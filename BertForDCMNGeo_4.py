from torch import nn
from transformers import BertPreTrainedModel, BertModel
import torch
from torch.autograd import Variable
import heapq
from torch.nn import CrossEntropyLoss


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear1(q)
        lp = self.linear2(p)
        mid = nn.Sigmoid()(lq + lp)
        output = p * mid + q * (1 - mid)
        return output


class SingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SingleMatchNet, self).__init__()
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)
        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.dense(att_vec))
        return output


class AOI(nn.Module):
    def __init__(self, config):
        super(AOI, self).__init__()
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.fusechoice = FuseNet(config)
        self.choice_match = SingleMatchNet(config)

    def forward(self, p, option_len):
        oc_seq = p.new(p.size()).zero_()
        for i in range(p.size(0)):
            op = p[i]
            if i % 4 == 0:
                op1 = p[i + 1]
                op2 = p[i + 2]
                op3 = p[i + 3]
                indx = [i + 1, i + 2, i + 3]
            elif i % 4 == 1:
                op1 = p[i - 1]
                op2 = p[i + 1]
                op3 = p[i + 2]
                indx = [i - 1, i + 1, i + 2]
            elif i % 4 == 2:
                op1 = p[i - 2]
                op2 = p[i - 1]
                op3 = p[i + 1]
                indx = [i - 2, i - 1, i + 1]
            else:
                op1 = p[i - 3]
                op2 = p[i - 2]
                op3 = p[i - 1]
                indx = [i - 3, i - 2, i - 1]
            # oc1 = self.get_choice_interaction([op1.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            # oc2 = self.get_choice_interaction([op2.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            # oc3 = self.get_choice_interaction([op3.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            oc1 = self.choice_match([op.unsqueeze(0), op1.unsqueeze(0), option_len[indx[0]].unsqueeze(0) + 1])
            oc2 = self.choice_match([op.unsqueeze(0), op2.unsqueeze(0), option_len[indx[1]].unsqueeze(0) + 1])
            oc3 = self.choice_match([op.unsqueeze(0), op3.unsqueeze(0), option_len[indx[2]].unsqueeze(0) + 1])
            cat_oc = torch.cat([oc1, oc2, oc3], 2)
            oc = self.dense3(cat_oc)
            oc_seq[i] = self.fusechoice([op, oc])
        return oc_seq


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    # doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    # ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        # doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, 1:doc_len[i] + ques_len[i] + 1]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + 1]
        # ques_option_seq_output[i, :ques_len[i] + option_len[i] + 1] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 2]
        option_seq_output[i, :option_len[i]] = sequence_output[i, doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[i] + 2]
    return doc_seq_output, ques_seq_output, option_seq_output


class BertForMultipleChoiceWithMatch(BertPreTrainedModel):

    def __init__(self, config, model_choices=300):
        super(BertForMultipleChoiceWithMatch, self).__init__(config)
        self.num_choices = 4
        self.model_choices = model_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier3 = nn.Linear(3 * config.hidden_size, 1)
        self.aoi = AOI(config)
        self.match_three = Match_Three(config)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, option_len=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()  # [1, bs*4]
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()  # [1, bs*4]
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()  # [1, bs*4]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        sequence_output, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(sequence_output, doc_len, ques_len, option_len)

        option_seq_output = self.aoi(option_seq_output, option_len)

        cat_pool = self.match_three(doc_seq_output, ques_seq_output, option_seq_output, doc_len, ques_len, option_len)

        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier3(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits


class BertForMC(BertPreTrainedModel):

    def __init__(self, config, model_choices=300):
        super().__init__(config)
        self.num_choices = 4
        self.model_choices = model_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, option_len=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        _, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        output_pool = self.dropout(pooled_output)
        match_logits = self.classifier(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits


class Match_Three(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.smatch = SingleMatchNet(config)
        self.fuse = FuseNet(config)

    def forward(self, doc_seq_output, ques_seq_output, option_seq_output, doc_len, ques_len, option_len):
        qp_output = self.smatch([ques_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.smatch([doc_seq_output, ques_seq_output, ques_len + 1])
        pa_output = self.smatch([doc_seq_output, option_seq_output, option_len + 1])
        ap_output = self.smatch([option_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.smatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.smatch([option_seq_output, ques_seq_output, ques_len + 1])
        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        return cat_pool
