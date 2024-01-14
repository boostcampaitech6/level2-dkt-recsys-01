import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # self.embedding_user_category = nn.Linear(1, intd).float()
        test_group_dim = 115
        self.embedding_test_group_one = nn.Embedding(1001, test_group_dim)
        self.embedding_test_group_two = nn.Embedding(1001, test_group_dim)

        serial_dim = 100
        self.embedding_serial = nn.Embedding(1001, serial_dim)

        # Concatentaed Embedding Projection
        features_len = intd * 4 + test_group_dim * 2 + serial_dim + 7
        self.comb_proj = nn.Linear(features_len, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
    def forward(self, 
                test, 
                question, 
                tag, 
                correct, 
                mask, 
                interaction, 
                duration, 
                test_group_one, 
                test_group_two, 
                serial, 
                solved_count, 
                correct_before, 
                wrong_before, 
                same_tag_solved_count,
                same_tag_correct_before,
                same_tag_wrong_before,
                ):
        # print(test.shape, question.shape, tag.shape, interaction.shape, duration.shape)
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        # embed_duration = self.embedding_duration(duration.unsqueeze(-1).float())
        # embed_user_category = self.embedding_user_category(user_category.unsqueeze(-1).float())
        embed_test_group_one = self.embedding_test_group_one(test_group_one.int())
        embed_test_group_two = self.embedding_test_group_one(test_group_two.int())
        embed_serial = self.embedding_serial(serial.int())

        # print(embed_tag.shape, embed_duration.shape)
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                duration.unsqueeze(-1).float(),
                embed_test_group_one,
                embed_test_group_two,
                # serial.unsqueeze(-1).int(),
                embed_serial,
                solved_count.unsqueeze(-1).int(),
                correct_before.unsqueeze(-1).int(),
                wrong_before.unsqueeze(-1).int(),
                same_tag_solved_count.unsqueeze(-1).int(),
                same_tag_correct_before.unsqueeze(-1).int(),
                same_tag_wrong_before.unsqueeze(-1).int(),
                # embed_user_category,
                # embed_time
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, 
                test, 
                question, 
                tag, 
                correct, 
                mask, 
                interaction,
                duration, 
                test_group_one, 
                test_group_two, 
                serial, 
                solved_count, 
                correct_before, 
                wrong_before,
                same_tag_solved_count,
                same_tag_correct_before,
                same_tag_wrong_before,
                ):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        duration=duration,
                                        test_group_one=test_group_one,
                                        test_group_two=test_group_two,
                                        serial=serial,
                                        solved_count=solved_count,
                                        correct_before=correct_before,
                                        wrong_before=wrong_before,
                                        same_tag_solved_count=same_tag_solved_count,
                                        same_tag_correct_before=same_tag_correct_before,
                                        same_tag_wrong_before=same_tag_wrong_before,
                                        )
        out, _ = self.lstm(X)
        out = out.contiguous()\
            .view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        out, _ = self.lstm(X)
        out = out.contiguous()\
            .view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
