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

        time_embedding_dim = 100
        self.embedding_time = nn.Linear(2, time_embedding_dim)

        # self.embedding_user_category = nn.Linear(1, intd).float()
        test_group_dim = 115
        self.embedding_test_group_one = nn.Embedding(1001, test_group_dim)
        self.embedding_test_group_two = nn.Embedding(1001, test_group_dim)

        serial_dim = 100
        self.embedding_serial = nn.Embedding(1001, serial_dim)

        correct_percent_dim = 50
        self.embedding_correct_percent = nn.Linear(2, correct_percent_dim)

        tag_group_dim = 50
        self.embedding_tag_group_one = nn.Embedding(n_tags * 1000 + 1, tag_group_dim)
        self.embedding_tag_group_two = nn.Embedding(n_tags * 1000 + 1, tag_group_dim)

        guess_dim = 10
        self.embedding_guess = nn.Embedding(3, guess_dim)
        self.embedding_guess_user = nn.Embedding(3, guess_dim)
        self.embedding_guess_test = nn.Embedding(3, guess_dim)
        self.embedding_guess_serial = nn.Embedding(3, guess_dim)
        self.embedding_guess_assessment = nn.Embedding(3, guess_dim)
        self.embedding_guess_tag = nn.Embedding(3, guess_dim)
        self.embedding_guess_day = nn.Embedding(3, guess_dim)
        self.embedding_guess_group_one = nn.Embedding(3, guess_dim)
        self.embedding_guess_group_two = nn.Embedding(3, guess_dim)

        

        # Concatentaed Embedding Projection3
        features_len = (intd * 4) + 1 + (test_group_dim * 2) + serial_dim + (tag_group_dim * 1) + 11 + (guess_dim * 1)
        
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
                startTime,
                elapsedTime,
                test_group_one,
                test_group_two,
                serial,
                solved_count,
                correct_before,
                wrong_before,
                same_tag_solved_count,
                same_tag_correct_before,
                same_tag_wrong_before,
                item_correct_percent,
                user_correct_percent,
                current_correct_count,
                tag_group_one,
                tag_group_two,
                time_for_solve,
                guess_yn,
                guess_yn_user,
                guess_yn_test,
                guess_yn_serial,
                guess_yn_assessment,
                guess_yn_tag,
                guess_yn_day,
                guess_yn_group_one,
                guess_yn_group_two,
                correct_percent_group_one,
                correct_percent_group_two,
                correct_percent_serial,
                day_of_week,
                duration_user,
                item_difficulty,
                ):
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed_test_group_one = self.embedding_test_group_one(test_group_one.int())
        embed_test_group_two = self.embedding_test_group_one(test_group_two.int())
        embed_serial = self.embedding_serial(serial.int())

        embed_tag_group_one = self.embedding_tag_group_one(tag_group_one.int())
        embed_tag_group_two = self.embedding_tag_group_two(tag_group_two.int())

        embed_guess = self.embedding_guess(guess_yn.int())
        embed_guess_user = self.embedding_guess_user(guess_yn_user.int())
        embed_guess_test = self.embedding_guess_test(guess_yn_test.int())
        embed_guess_serial = self.embedding_guess_serial(guess_yn_serial.int())
        embed_guess_assessment = self.embedding_guess_assessment(guess_yn_assessment.int())
        embed_guess_tag = self.embedding_guess_tag(guess_yn_tag.int())
        embed_guess_day = self.embedding_guess_day(guess_yn_day.int())
        embed_guess_group_one = self.embedding_guess_group_one(guess_yn_group_one.int())
        embed_guess_group_two = self.embedding_guess_group_two(guess_yn_group_two.int())

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                duration.unsqueeze(-1).float(),
                embed_test_group_one,
                embed_test_group_two,
                embed_serial,
                solved_count.unsqueeze(-1).float(),
                correct_before.unsqueeze(-1).float(),
                wrong_before.unsqueeze(-1).float(),
                same_tag_solved_count.unsqueeze(-1).float(),
                same_tag_correct_before.unsqueeze(-1).float(),
                same_tag_wrong_before.unsqueeze(-1).float(),
                current_correct_count.unsqueeze(-1).float(),
                # embed_tag_group_one,
                embed_tag_group_two,
                time_for_solve.unsqueeze(-1).float(),
                user_correct_percent.unsqueeze(-1).float(),
                # item_correct_percent.unsqueeze(-1).float(),
                embed_guess,
                # guess_yn_user.unsqueeze(-1).float(),
                # guess_yn_test.unsqueeze(-1).float(),
                # guess_yn_serial.unsqueeze(-1).float(),
                # guess_yn_assessment.unsqueeze(-1).float(),
                # guess_yn_tag.unsqueeze(-1).float(),
                # guess_yn_day.unsqueeze(-1).float(),
                # guess_yn_group_one.unsqueeze(-1).float(),
                # guess_yn_group_two.unsqueeze(-1).float(),
                # embed_guess_user,
                # embed_guess_test,
                # embed_guess_serial,
                # embed_guess_assessment,
                # embed_guess_tag,
                # embed_guess_day,
                # embed_guess_group_one,
                # embed_guess_group_two,
                day_of_week.unsqueeze(-1).int(),
                item_difficulty.unsqueeze(-1).float(),
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
                # assessmentItemID,
                question, 
                tag, 
                correct,
                mask, 
                interaction,
                duration,
                startTime,
                elapsedTime,
                test_group_one, 
                test_group_two, 
                serial, 
                solved_count, 
                correct_before, 
                wrong_before,
                same_tag_solved_count,
                same_tag_correct_before,
                same_tag_wrong_before,
                item_correct_percent,
                user_correct_percent,
                current_correct_count,
                tag_group_one,
                tag_group_two,
                time_for_solve,
                guess_yn,
                guess_yn_user,
                guess_yn_test,
                guess_yn_serial,
                guess_yn_assessment,
                guess_yn_tag,
                guess_yn_day,
                guess_yn_group_one,
                guess_yn_group_two,
                correct_percent_group_one,
                correct_percent_group_two,
                correct_percent_serial,
                day_of_week,
                duration_user,
                item_difficulty,
                ):
        X, batch_size = super().forward(test=test,
                                        # assessmentItemID=assessmentItemID,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction,
                                        duration=duration,
                                        startTime=startTime,
                                        elapsedTime=elapsedTime,
                                        test_group_one=test_group_one,
                                        test_group_two=test_group_two,
                                        serial=serial,
                                        solved_count=solved_count,
                                        correct_before=correct_before,
                                        wrong_before=wrong_before,
                                        same_tag_solved_count=same_tag_solved_count,
                                        same_tag_correct_before=same_tag_correct_before,
                                        same_tag_wrong_before=same_tag_wrong_before,
                                        item_correct_percent=item_correct_percent,
                                        user_correct_percent=user_correct_percent,
                                        current_correct_count=current_correct_count,
                                        tag_group_one=tag_group_one,
                                        tag_group_two=tag_group_two,
                                        time_for_solve=time_for_solve,
                                        guess_yn=guess_yn,
                                        guess_yn_user=guess_yn_user,
                                        guess_yn_test=guess_yn_test,
                                        guess_yn_serial=guess_yn_serial,
                                        guess_yn_assessment=guess_yn_assessment,
                                        guess_yn_tag=guess_yn_tag,
                                        guess_yn_day=guess_yn_day,
                                        guess_yn_group_one=guess_yn_group_one,
                                        guess_yn_group_two=guess_yn_group_two,
                                        correct_percent_group_one=correct_percent_group_one,
                                        correct_percent_group_two=correct_percent_group_two,
                                        correct_percent_serial=correct_percent_serial,
                                        day_of_week=day_of_week,
                                        duration_user=duration_user,
                                        item_difficulty=item_difficulty,
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
