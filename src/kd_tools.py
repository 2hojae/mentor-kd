import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaKDLoss(nn.Module):
    def __init__(self, temperature=2.0, eps=1e-8):
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, student_logits, teacher_logits):
        p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_student, p_teacher)        
        return loss
    