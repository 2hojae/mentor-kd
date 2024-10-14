import collections
import os
from logging import getLogger
from pathlib import Path
from typing import List

import torch

from src.evaluator import DistributedEvaluator
from src.model import DistributedModule
from src.tokenizer import Tokenizer
from src.utils import cross_entropy, barrier, kl_div_loss_v2, kl_div_loss

logger = getLogger()


class DistributedTrainer:
    def __init__(self,
                 model: DistributedModule,
                 tokenizer: Tokenizer,
                 optimizer: torch.optim.Optimizer,
                 eval_batch_size: int,
                 accumulation_steps: int = 1,
                 log_dir: str = "log/"):
        self.model = model
        #self.local_rank = model.local_rank
        self.local_rank = 0
        
        #self.world_size = model.world_size
        #self.max_seq_len = self.model.params.max_seq_len
        self.tokenizer = tokenizer
        self.max_seq_len = 512
        self.optimizer = optimizer
        self.evaluator = DistributedEvaluator(self.model, tokenizer)
        self.step = 0
        self.accumulation_steps = accumulation_steps
        self.eval_batch_size = eval_batch_size
        self.log_dir = log_dir

    def __truncating_strategy(self, instruction_ids, output_ids):
        instruction_length = len(instruction_ids)
        output_length = len(output_ids)
        if instruction_length >= self.max_seq_len:
            print(f'WARNING: Length of instruction {instruction_length} '
                  f'exceeds the max input length {self.max_seq_len}')
            instruction_ids = instruction_ids[:self.max_seq_len]
            instruction_length = len(instruction_ids)
        sequence_length = instruction_length + output_length
        if sequence_length > self.max_seq_len:
            exceed_length = sequence_length - self.max_seq_len
            output_ids = output_ids[:-exceed_length]
        return instruction_ids, output_ids

    def __back_propagation(self, loss: torch.Tensor):
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def __prepare_for_training(self, instructions, outputs):
        """ :return tokens, labels, label_mask,  """
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long().cuda()
        labels = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long().cuda()
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            instruction_ids, output_ids = self.__truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
        label_mask = (labels != self.tokenizer.pad_id)
        Output = collections.namedtuple('Outputs', ['tokens', 'labels', 'label_mask'])
        return Output(tokens=tokens, labels=labels, label_mask=label_mask)

    @torch.no_grad()
    def predict(self, logits, instructions: List[str], outputs: List[str]) -> List[dict]:
        bzs = int(logits.shape[0])
        datalist = []
        for i in range(bzs):
            instruction_ids = self.tokenizer.tokenize(instructions[i], bos=True)
            output_ids = self.tokenizer.tokenize(outputs[i], eos=True)
            instruction_ids, output_ids = self.__truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            predict_ids = torch.argmax(logits[i], dim=-1)[instr_len - 1: instr_len - 1 + output_len].tolist()
            datalist.append(dict(instruction=instructions[i], output=self.tokenizer.decode(predict_ids)))
        return datalist

    def train(self, instructions: List[str], outputs: List[str]):
        """ Instruction tuning """
        example = self.__prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens)
        loss = cross_entropy(logits, example.labels, example.label_mask.to(logits.device))
        self.__back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits'])
        return Output(logits=logits, loss=loss)

    def __prepare_for_indices(self, indices1: list, indices2: list):
        ids1_list = []
        ids2_list = []
        maxlen = 0
        for ids1, ids2 in zip(indices1, indices2):
            ids1 = str(ids1).split(' ')
            ids2 = str(ids2).split(' ')
            assert len(ids1) == len(ids2)
            _ids1 = []
            _ids2 = []
            for i in range(len(ids1) // 2):
                if int(ids1[i*2+1]) >= self.max_seq_len or int(ids2[i*2+1]) >= self.max_seq_len:
                    _ids1.append([0, 0])
                    _ids2.append([0, 0])
                else:
                    _ids1.append([int(ids1[i*2]), int(ids1[i*2+1])])
                    _ids2.append([int(ids2[i*2]), int(ids2[i*2+1])])
            maxlen = max(maxlen, len(_ids1))
            ids1_list.append(_ids1)
            ids2_list.append(_ids2)
        for ids1, ids2 in zip(ids1_list, ids2_list):
            while len(ids1) < maxlen:
                ids1.append([0, 0])
                ids2.append([0, 0])
        ids1_list, ids2_list = torch.tensor(ids1_list).transpose(1, 0), torch.tensor(ids2_list).transpose(1, 0)
        return ids1_list, ids2_list

    def __compute_indexing_kl_div_loss(
            self,
            logits1: torch.Tensor,
            logits2: torch.Tensor,
            indices1: list,
            indices2: list
    ):
        kl_loss = torch.tensor(0.0)
        indices1, indices2 = self.__prepare_for_indices(indices1, indices2)
        for t, (ids1, ids2) in enumerate(zip(indices1, indices2)):
            bzs = ids1.shape[0]
            vocab_size = logits1.shape[-1]
            max_len1 = max(torch.sub(ids1[:, 1], ids1[:, 0]))
            max_len2 = max(torch.sub(ids2[:, 1], ids2[:, 0]))
            assert max_len1 == max_len2
            P = torch.full(size=(bzs, max_len1, vocab_size), fill_value=0.).float()
            Q = torch.full(size=(bzs, max_len2, vocab_size), fill_value=0.).float()
            for i in range(bzs):
                P[i, : ids1[i, 1] - ids1[i, 0], :] = logits1[i, ids1[i, 0]: ids1[i, 1], :]
                Q[i, : ids2[i, 1] - ids2[i, 0], :] = logits2[i, ids2[i, 0]: ids2[i, 1], :]
            M = (torch.sum(P, dim=-1) != 0)
            P = torch.softmax(P, dim=-1)
            Q = torch.softmax(Q, dim=-1)
            kl_loss = (kl_loss * t + kl_div_loss_v2(P, Q, weights=M)) / (t + 1)
        return kl_loss

    def train_mcc(self,
                  instructions: List[str],
                  outputs1: List[str],
                  outputs2: List[str],
                  indices1,
                  indices2,
                  alpha: float = 1.0,
                  temperature: float = 1.0
                  ):
        example1 = self.__prepare_for_training(instructions=instructions, outputs=outputs1)
        example2 = self.__prepare_for_training(instructions=instructions, outputs=outputs2)
        logits1 = self.model.forward(example1.tokens)
        logits2 = self.model.forward(example2.tokens)
        ce_loss1 = cross_entropy(logits1, example1.labels, example1.label_mask.to(logits1.device))
        ce_loss2 = cross_entropy(logits2, example2.labels, example2.label_mask.to(logits2.device))
        ce_loss = (ce_loss1 + ce_loss2) * 0.5

        # compute kl div loss
        indices1, indices2 = indices1[0], indices2[0]
        indices1 = torch.cat([t.unsqueeze(0) for t in indices1], dim=0).to(logits1.device).T
        indices2 = torch.cat([t.unsqueeze(0) for t in indices2], dim=0).to(logits2.device).T
        bzs = indices1.shape[0]
        vocab_size = logits1.shape[-1]
        max_len1 = max(torch.sub(indices1[:, 1], indices1[:, 0]))
        max_len2 = max(torch.sub(indices2[:, 1], indices2[:, 0]))
        assert max_len1 == max_len2
        P = torch.full(size=(bzs, max_len1, vocab_size), fill_value=0.).float()
        Q = torch.full(size=(bzs, max_len2, vocab_size), fill_value=0.).float()
        unexceeded_batch_indices = []  # only count for those within `max_seq_len`
        for i in range(bzs):
            if indices1[i, 1] >= self.max_seq_len or indices2[i, 1] >= self.max_seq_len:
                print(f'WARNING: Escaping batch index because {max(indices1[i, 1], indices2[i, 1])} '
                      f'exceeding max length {self.max_seq_len}')
                continue
            P[i, : indices1[i, 1] - indices1[i, 0], :] = logits1[i, indices1[i, 0]: indices1[i, 1], :]
            Q[i, : indices2[i, 1] - indices2[i, 0], :] = logits2[i, indices2[i, 0]: indices2[i, 1], :]
            unexceeded_batch_indices.append(i)
        if len(unexceeded_batch_indices) > 0:
            P = P[unexceeded_batch_indices]
            Q = Q[unexceeded_batch_indices]
            M = (torch.sum(P, dim=-1) != 0)
            P = torch.softmax(P / temperature, dim=-1)
            Q = torch.softmax(Q / temperature, dim=-1)
            kl_loss = kl_div_loss(P, Q, weights=M)
        else:
            kl_loss = torch.tensor(0.0)

        loss = ce_loss + alpha * kl_loss
        self.__back_propagation(loss)

        # For evaluation
        tokens1 = self.tokenizer.detokenize(example1.tokens[0, indices1[0, 0]: indices1[0, 1]].tolist())
        tokens2 = self.tokenizer.detokenize(example2.tokens[0, indices2[0, 0]: indices2[0, 1]].tolist())
        Output = collections.namedtuple(
            'Output', ['logits1', 'logits2', 'ce_loss', 'kl_loss', 'info'])
        return Output(logits1=logits1,
                      logits2=logits2,
                      ce_loss=ce_loss,
                      kl_loss=kl_loss,
                      info=[tokens1, tokens2])

    def train_with_consisting(
            self,
            instructions: List[str],
            outputs1: List[str],
            outputs2: List[str],
            indices1,
            indices2,
            alpha: float = 1.0
    ):
        example1 = self.__prepare_for_training(instructions=instructions, outputs=outputs1)
        example2 = self.__prepare_for_training(instructions=instructions, outputs=outputs2)
        logits1 = self.model.forward(example1.tokens)
        logits2 = self.model.forward(example2.tokens)
        ce_loss1 = cross_entropy(logits1, example1.labels, example1.label_mask.to(logits1.device))
        ce_loss2 = cross_entropy(logits2, example2.labels, example2.label_mask.to(logits2.device))
        ce_loss = (ce_loss1 + ce_loss2) * 0.5

        # ind1, ind2 = self.__prepare_for_indices(indices1, indices2)
        # for t, (ids1, ids2) in enumerate(zip(ind1, ind2)):
        #     print(self.tokenizer.detokenize(example1.tokens[0][ids1[0, 0]: ids1[0, 1]].tolist()))
        #     print(self.tokenizer.detokenize(example2.tokens[0][ids2[0, 0]: ids2[0, 1]].tolist()))

        kl_loss = self.__compute_indexing_kl_div_loss(logits1, logits2, indices1, indices2)

        loss = ce_loss + alpha * kl_loss
        self.__back_propagation(loss)

        Output = collections.namedtuple(
            'Output', ['logits1', 'logits2', 'ce_loss', 'kl_loss'])
        return Output(logits1=logits1,
                      logits2=logits2,
                      ce_loss=ce_loss,
                      kl_loss=kl_loss)

    def evaluate(self,
                 task: str,
                 label_file,
                 output_file,
                 max_seq_len=512):
        if not os.path.exists(self.log_dir) and self.local_rank == 0:
            os.makedirs(self.log_dir)
        #barrier()
        output_file = os.path.join(self.log_dir, output_file)
        accuracy = self.evaluator.evaluate(
            task=task,
            label_file=label_file,
            output_file=output_file,
            batch_size=self.eval_batch_size,
            max_seq_len=max_seq_len,
            temperature=0.0,  # TODO random seed can not control when temperature > 0.0,
            top_p=0.0
        )
        return accuracy

    def save_distributed_optimizer(self, save_path: str):
        if not os.path.exists(save_path) and self.local_rank == 0:
            os.makedirs(save_path)
        print(f'Saving optimizer to {save_path} ......')
        #barrier()
        torch.save(self.optimizer.state_dict(), os.path.join(
            save_path, f'optimizer.0{self.local_rank}.bin'))
        #barrier()
        print(f'Saving done !')

    def load_distributed_optimizer(self, save_path: str):
        checkpoints = sorted(Path(save_path).glob("optimizer.*.bin"))
        if len(checkpoints) == 0:
            return
        print(f'Loading optimizer from {save_path} .....')
        '''
        assert self.world_size == len(
            checkpoints
        ), f"Loading a optimizer for MP={len(checkpoints)} but world size is {self.world_size}"
        '''
        optim_file = checkpoints[self.local_rank]
        state_dict = torch.load(optim_file)
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f'Loading done !')

    def save_distributed_model(self, save_path: str):
        self.model.save(save_path)

    def load_distributed_model(self, save_path: str):
        self.model.load(save_path)

    def load(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not loading model because `save_path` is None")
            return
        self.load_distributed_optimizer(save_path)
        self.load_distributed_model(save_path)

    def save(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not saving model because `save_path` is None")
            return
        self.save_distributed_optimizer(save_path)
        self.save_distributed_model(save_path)


class DistributedTrainer_T5:
    def __init__(self,
                 model,
                 tokenizer: Tokenizer,
                 optimizer: torch.optim.Optimizer,
                 eval_batch_size: int,
                 accumulation_steps: int = 1,
                 log_dir: str = "log/"):
        self.model = model
        #self.local_rank = model.local_rank
        self.local_rank = 0
        
        #self.world_size = model.world_size
        #self.max_seq_len = self.model.params.max_seq_len

        self.tokenizer = tokenizer
        self.max_seq_len = 512
        self.optimizer = optimizer
        self.evaluator = DistributedEvaluator(self.model, tokenizer)
        self.step = 0
        self.accumulation_steps = accumulation_steps
        self.eval_batch_size = eval_batch_size
        self.log_dir = log_dir

    def __truncating_strategy(self, instruction_ids, output_ids):
        instruction_length = len(instruction_ids)
        output_length = len(output_ids)
        if instruction_length >= self.max_seq_len:
            print(f'WARNING: Length of instruction {instruction_length} '
                  f'exceeds the max input length {self.max_seq_len}')
            instruction_ids = instruction_ids[:self.max_seq_len]
            instruction_length = len(instruction_ids)
        if output_length >= self.max_seq_len:
            output_ids = output_ids[:self.max_seq_len]
        '''
        sequence_length = instruction_length + output_length
        if sequence_length > self.max_seq_len:
            exceed_length = sequence_length - self.max_seq_len
            output_ids = output_ids[:-exceed_length]
        '''
        return instruction_ids, output_ids

    def __back_propagation(self, loss: torch.Tensor):
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def __prepare_for_training(self, instructions, outputs):
        """ :return tokens, labels, label_mask,  """
        bsz = len(instructions)
        
        #print(outputs, len(outputs))
        
        #print(instructions, outputs)
        
        t_lengths = [self.tokenizer.encode(s, return_tensors = 'pt').shape[1] for s in instructions]
        l_lengths = [self.tokenizer.encode(s, return_tensors = 'pt').shape[1] for s in outputs]
        
        #print(instructions, outputs)
        
        max_t_lengths = min(max(t_lengths), 512)
        max_l_lengths = min(max(l_lengths), 512)
        
        #tokens = torch.full((bsz, max(t_lengths)), self.tokenizer.pad_token_id).long().cuda()
        #labels = torch.full((bsz, max(l_lengths)), self.tokenizer.pad_token_id).long().cuda()
        tokens = torch.full((bsz, max_t_lengths), self.tokenizer.pad_token_id).long().cuda()
        labels = torch.full((bsz, max_l_lengths), self.tokenizer.pad_token_id).long().cuda()
        
        #print(instructions, outputs)
        
        #tokens = self.tokenizer(instructions, padding = 'longest', max_length = 128, truncation = True, return_tensors = 'pt')['input_ids'].cuda()
        #labels = self.tokenizer(outputs, padding = 'longest', max_length = 384, truncation = True, return_tensors = 'pt')['input_ids'].cuda()
        
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            #instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            #output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            
            #print(instruction, output)
            instruction_ids = self.tokenizer.encode(instruction)#, return_tensors = 'pt')
            #output_ids = self.tokenizer.encode(output[0])       
            #print(output)     
            #if train with train.json -> output[0] in normal training
            #print(output)
            output_ids = self.tokenizer.encode(output)#, return_tensors = 'pt')
            
            #print('preprocessing; ' , len(output_ids) , max_l_lengths)
            
            instruction_ids, output_ids = self.__truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            #tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            #labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
            
            #print('preprocessed2; ' , len(output_ids) , max_l_lengths)
            
            tokens[i, :instr_len] = torch.tensor(instruction_ids).long().cuda()
            labels[i,:output_len] = torch.tensor(output_ids).long().cuda()
        
        
        label_mask = (labels != self.tokenizer.pad_token_id).cuda()
        labels[~label_mask.bool()] = -100
        Output = collections.namedtuple('Outputs', ['tokens', 'labels', 'label_mask'])
        return Output(tokens=tokens, labels=labels, label_mask=label_mask)

    @torch.no_grad()
    def predict(self, logits, instructions: List[str], outputs: List[str]) -> List[dict]:
        bzs = int(logits.shape[0])
        datalist = []
        for i in range(bzs):
            #instruction_ids = self.tokenizer.tokenize(instructions[i], bos=True)
            #output_ids = self.tokenizer.tokenize(outputs[i], eos=True)
            instruction_ids = self.tokenizer.tokenize(instructions[i], return_tensors = 'pt')#.cuda()
            output_ids = self.tokenizer.tokenize(outputs[i], return_tensors = 'pt')#.cuda()
            
            #print(instruction_ids, output_ids)
            
            instruction_ids, output_ids = self.__truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            #predict_ids = torch.argmax(logits[i], dim=-1)[instr_len - 1: instr_len - 1 + output_len].tolist()
            predict_ids = torch.argmax(logits[i], dim=-1).tolist()
            
            datalist.append(dict(instruction=instructions[i], output=self.tokenizer.decode(predict_ids)))
        return datalist

    def train(self, instructions: List[str], outputs: List[str]):
        """ Instruction tuning """
        example = self.__prepare_for_training(instructions=instructions, outputs=outputs)
        #logits = self.model.forward(example.tokens)
        #loss = cross_entropy(logits, example.labels, example.label_mask.to(logits.device))
        outputs = self.model(input_ids = example.tokens, labels = example.labels, decoder_attention_mask = example.label_mask)
        loss = outputs.loss
        logits = outputs.logits
        
        self.__back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits'])
        return Output(logits=logits, loss=loss)

    def __prepare_for_indices(self, indices1: list, indices2: list):
        ids1_list = []
        ids2_list = []
        maxlen = 0
        for ids1, ids2 in zip(indices1, indices2):
            ids1 = str(ids1).split(' ')
            ids2 = str(ids2).split(' ')
            assert len(ids1) == len(ids2)
            _ids1 = []
            _ids2 = []
            for i in range(len(ids1) // 2):
                if int(ids1[i*2+1]) >= self.max_seq_len or int(ids2[i*2+1]) >= self.max_seq_len:
                    _ids1.append([0, 0])
                    _ids2.append([0, 0])
                else:
                    _ids1.append([int(ids1[i*2]), int(ids1[i*2+1])])
                    _ids2.append([int(ids2[i*2]), int(ids2[i*2+1])])
            maxlen = max(maxlen, len(_ids1))
            ids1_list.append(_ids1)
            ids2_list.append(_ids2)
        for ids1, ids2 in zip(ids1_list, ids2_list):
            while len(ids1) < maxlen:
                ids1.append([0, 0])
                ids2.append([0, 0])
        ids1_list, ids2_list = torch.tensor(ids1_list).transpose(1, 0), torch.tensor(ids2_list).transpose(1, 0)
        return ids1_list, ids2_list

    def __compute_indexing_kl_div_loss(
            self,
            logits1: torch.Tensor,
            logits2: torch.Tensor,
            indices1: list,
            indices2: list
    ):
        kl_loss = torch.tensor(0.0)
        indices1, indices2 = self.__prepare_for_indices(indices1, indices2)
        for t, (ids1, ids2) in enumerate(zip(indices1, indices2)):
            bzs = ids1.shape[0]
            vocab_size = logits1.shape[-1]
            max_len1 = max(torch.sub(ids1[:, 1], ids1[:, 0]))
            max_len2 = max(torch.sub(ids2[:, 1], ids2[:, 0]))
            assert max_len1 == max_len2
            P = torch.full(size=(bzs, max_len1, vocab_size), fill_value=0.).float()
            Q = torch.full(size=(bzs, max_len2, vocab_size), fill_value=0.).float()
            for i in range(bzs):
                P[i, : ids1[i, 1] - ids1[i, 0], :] = logits1[i, ids1[i, 0]: ids1[i, 1], :]
                Q[i, : ids2[i, 1] - ids2[i, 0], :] = logits2[i, ids2[i, 0]: ids2[i, 1], :]
            M = (torch.sum(P, dim=-1) != 0)
            P = torch.softmax(P, dim=-1)
            Q = torch.softmax(Q, dim=-1)
            kl_loss = (kl_loss * t + kl_div_loss_v2(P, Q, weights=M)) / (t + 1)
        return kl_loss

    def train_mcc(self,
                  instructions: List[str],
                  outputs1: List[str],
                  outputs2: List[str],
                  indices1,
                  indices2,
                  alpha: float = 1.0,
                  temperature: float = 1.0
                  ):
        example1 = self.__prepare_for_training(instructions=instructions, outputs=outputs1)
        example2 = self.__prepare_for_training(instructions=instructions, outputs=outputs2)
        #logits1 = self.model.forward(example1.tokens)
        #logits2 = self.model.forward(example2.tokens)
        '''
        print(indices1, indices2)
        i = 0
        for o1,o2 in zip(outputs1, outputs2):
            o1 = self.tokenizer.encode(o1)
            o2 = self.tokenizer.encode(o2)
            
            o1 = self.tokenizer.decode(o1[indices1[i][0] : indices1[i][1]])
            o2 = self.tokenizer.decode(o2[indices2[i][0] : indices2[i][1]])
            print(o1,o2)
            i += 1
        '''
        outputs1 = self.model(input_ids = example1.tokens, labels = example1.labels, decoder_attention_mask = example1.label_mask)
        outputs2 = self.model(input_ids = example2.tokens, labels = example2.labels, decoder_attention_mask = example2.label_mask)
        
        logits1 = outputs1.logits
        logits2 = outputs2.logits
        
        
        #print(example1.labels.shape, logits1.shape)
        #print(example2.labels.shape, logits2.shape)
        #print(indices1, indices2)
        #ce_loss1 = cross_entropy(logits1, example1.labels, example1.label_mask.to(logits1.device))
        #ce_loss2 = cross_entropy(logits2, example2.labels, example2.label_mask.to(logits2.device))
        ce_loss1 = outputs1.loss
        ce_loss2 = outputs2.loss
        
        ce_loss = (ce_loss1 + ce_loss2) * 0.5

        # compute kl div loss
        #indices1, indices2 = indices1[0], indices2[0]
        #indices1 = torch.cat([t.unsqueeze(0) for t in indices1], dim=0).to(logits1.device).T
        #indices2 = torch.cat([t.unsqueeze(0) for t in indices2], dim=0).to(logits2.device).T
        indices1 = torch.Tensor(indices1).to(logits1.device).to(torch.long)   # (B, 2)
        indices2 = torch.Tensor(indices2).to(logits2.device).to(torch.long)   # (B, 2)
        
        #print(indices1, indices2)
        bzs = indices1.shape[0]   # batch size
        vocab_size = logits1.shape[-1]
        max_len1 = max(torch.sub(indices1[:, 1], indices1[:, 0]))   # maxlen of indices
        max_len2 = max(torch.sub(indices2[:, 1], indices2[:, 0]))
        assert max_len1 == max_len2
        P = torch.full(size=(bzs, max_len1, vocab_size), fill_value=0.).float()
        Q = torch.full(size=(bzs, max_len2, vocab_size), fill_value=0.).float()
        unexceeded_batch_indices = []  # only count for those within `max_seq_len`
        
        #print(logits1.shape, logits2.shape, indices1, indices2)
        for i in range(bzs):
            if indices1[i, 1] >= self.max_seq_len or indices2[i, 1] >= self.max_seq_len:
                print(f'WARNING: Escaping batch index because {max(indices1[i, 1], indices2[i, 1])} '
                      f'exceeding max length {self.max_seq_len}')
                continue
            #print(indices1[i], indices2[i])
            # print(logits1[i, indices1[i, 0]: indices1[i, 1], :].shape, logits2[i, indices2[i, 0]: indices2[i, 1], :].shape)
            
            # print(P.shape, Q.shape)   # both (B, max_len1, V=32128)
            # exit()
            # print(indices1.shape, indices2.shape)   # (B, 2)
            # print(logits1.shape, logits2.shape)   # (B, 163, V), (B, 74, V)
            # print(P[i, : indices1[i, 1] - indices1[i, 0], :].shape, logits1[i, indices1[i, 0]: indices1[i, 1], :].shape)   # (B, V)
            # print(logits1[i, indices1[i, 0]: indices1[i, 1], :].shape)   # (0, V)
            # print(P.shape, logits1.shape)
            # print(indices1[i, 1], indices1[i, 0])
            # a = indices1[i, 1] - indices1[i, 0]
            # print(a)
            # exit()
            
            P[i, : indices1[i, 1] - indices1[i, 0], :] = logits1[i, indices1[i, 0]: indices1[i, 1], :]
            Q[i, : indices2[i, 1] - indices2[i, 0], :] = logits2[i, indices2[i, 0]: indices2[i, 1], :]
            unexceeded_batch_indices.append(i)
        if len(unexceeded_batch_indices) > 0:
            P = P[unexceeded_batch_indices]
            Q = Q[unexceeded_batch_indices]
            M = (torch.sum(P, dim=-1) != 0)
            P = torch.softmax(P / temperature, dim=-1)
            Q = torch.softmax(Q / temperature, dim=-1)
            kl_loss = kl_div_loss(P, Q, weights=M)
        else:
            kl_loss = torch.tensor(0.0)

        loss = ce_loss + alpha * kl_loss
        self.__back_propagation(loss)

        # For evaluation
        #tokens1 = self.tokenizer.detokenize(example1.tokens[0, indices1[0, 0]: indices1[0, 1]].tolist())
        #tokens2 = self.tokenizer.detokenize(example2.tokens[0, indices2[0, 0]: indices2[0, 1]].tolist())
        #print(indices1, indices2, example1.labels.shape, example2.labels.shape)
        #print(example1.labels, example2.labels)
        
        t1 = example1.labels[0, indices1[0, 0]: indices1[0, 1]].tolist()
        t2 = example2.labels[0, indices2[0, 0]: indices2[0, 1]].tolist()
        
        #print(t1, t2)
        #print(indices1, indices2)
        #print(example1.labels, example2.labels)
        tokens1 = self.tokenizer.decode(t1)
        tokens2 = self.tokenizer.decode(t2)

        #print(tokens1, tokens2)

        Output = collections.namedtuple(
            'Output', ['logits1', 'logits2', 'ce_loss', 'kl_loss', 'info'])
        return Output(logits1=logits1,
                      logits2=logits2,
                      ce_loss=ce_loss,
                      kl_loss=kl_loss,
                      info=[tokens1, tokens2])

    def train_with_consisting(
            self,
            instructions: List[str],
            outputs1: List[str],
            outputs2: List[str],
            indices1,
            indices2,
            alpha: float = 1.0
    ):
        example1 = self.__prepare_for_training(instructions=instructions, outputs=outputs1)
        example2 = self.__prepare_for_training(instructions=instructions, outputs=outputs2)
        logits1 = self.model.forward(example1.tokens)
        logits2 = self.model.forward(example2.tokens)
        ce_loss1 = cross_entropy(logits1, example1.labels, example1.label_mask.to(logits1.device))
        ce_loss2 = cross_entropy(logits2, example2.labels, example2.label_mask.to(logits2.device))
        ce_loss = (ce_loss1 + ce_loss2) * 0.5

        # ind1, ind2 = self.__prepare_for_indices(indices1, indices2)
        # for t, (ids1, ids2) in enumerate(zip(ind1, ind2)):
        #     print(self.tokenizer.detokenize(example1.tokens[0][ids1[0, 0]: ids1[0, 1]].tolist()))
        #     print(self.tokenizer.detokenize(example2.tokens[0][ids2[0, 0]: ids2[0, 1]].tolist()))

        kl_loss = self.__compute_indexing_kl_div_loss(logits1, logits2, indices1, indices2)

        loss = ce_loss + alpha * kl_loss
        self.__back_propagation(loss)

        Output = collections.namedtuple(
            'Output', ['logits1', 'logits2', 'ce_loss', 'kl_loss'])
        return Output(logits1=logits1,
                      logits2=logits2,
                      ce_loss=ce_loss,
                      kl_loss=kl_loss)

    def evaluate(self,
                 task: str,
                 label_file,
                 output_file,
                 max_seq_len=512):
        if not os.path.exists(self.log_dir) and self.local_rank == 0:
            os.makedirs(self.log_dir)
        #barrier()
        output_file = os.path.join(self.log_dir, output_file)
        accuracy = self.evaluator.evaluate(
            task=task,
            label_file=label_file,
            output_file=output_file,
            batch_size=self.eval_batch_size,
            max_seq_len=max_seq_len,
            temperature=0.0,  # TODO random seed can not control when temperature > 0.0,
            top_p=0.0
        )
        
            
        return accuracy

    def save_distributed_optimizer(self, save_path: str):
        if not os.path.exists(save_path) and self.local_rank == 0:
            os.makedirs(save_path)
        print(f'Saving optimizer to {save_path} ......')
        #barrier()
        torch.save(self.optimizer.state_dict(), os.path.join(
            save_path, f'optimizer.0{self.local_rank}.bin'))
        #barrier()
        print(f'Saving done !')

    def load_distributed_optimizer(self, save_path: str):
        checkpoints = sorted(Path(save_path).glob("optimizer.*.bin"))
        if len(checkpoints) == 0:
            return
        print(f'Loading optimizer from {save_path} .....')
        '''
        assert self.world_size == len(
            checkpoints
        ), f"Loading a optimizer for MP={len(checkpoints)} but world size is {self.world_size}"
        '''
        optim_file = checkpoints[self.local_rank]
        state_dict = torch.load(optim_file)
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f'Loading done !')

    def save_distributed_model(self, save_path: str):
        #self.model.save(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), save_path + "best_model.pt")

    def load_distributed_model(self, save_path: str):
        #self.model.load(save_path)
        self.model.load_state_dict(torch.load(save_path + "best_model.pt"))

    def load(self, save_path: str):
        #self.load_distributed_optimizer(save_path)
        self.load_distributed_model(save_path)

    def save(self, save_path: str):
        #self.save_distributed_optimizer(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_distributed_model(save_path)
