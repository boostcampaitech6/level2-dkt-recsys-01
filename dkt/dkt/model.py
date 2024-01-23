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
        
        day_dim = 16
        self.embedding_day = nn.Embedding(7, day_dim)

        # Concatentaed Embedding Projection3
        features_len = (intd * 4) + 1 + (test_group_dim * 2) + serial_dim + (tag_group_dim * 1) + 11 + (guess_dim * 1) + day_dim + 18 + 136
        
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
                zero,
                user_ability,
                day_correct_percent,
                user_mode_hour,
                hour,
                year,
                user_mode_year,
                test_min_year,
                test_mode_year,
                test_max_year,
                item_min_year,
                item_mode_year,
                item_max_year,
                user_max_year,
                user_min_year,
                user_period_year,
                test_count,
                item_count,
                time_diff,
                user_solve_count,
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

        embed_day = self.embedding_day(day_of_week.int())

        dd = [test,
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
                zero,
                user_ability,
                day_correct_percent,
                user_mode_hour,
                hour,
                year,
                user_mode_year,
                test_min_year,
                test_mode_year,
                test_max_year,
                item_min_year,
                item_mode_year,
                item_max_year,
                user_max_year,
                user_min_year,
                user_period_year,
                test_count,
                item_count,
                time_diff,
                user_solve_count,
                ]

        # "test": torch.tensor(test + 1, dtype=torch.int),
        #     "question": torch.tensor(question + 1, dtype=torch.int),
        #     "tag": torch.tensor(tag + 1, dtype=torch.int),
        #     "correct": torch.tensor(correct, dtype=torch.int),

        #     "duration": torch.tensor(duration, dtype=torch.float),
        #     "startTime": torch.tensor(startTime, dtype=torch.float),
        #     "elapsedTime": torch.tensor(elapsedTime, dtype=torch.float),
        #     "test_group_one": torch.tensor(testGroupOne + 1, dtype=torch.int),

        #     "test_group_two": torch.tensor(testGroupTwo + 1, dtype=torch.int),
        #     "serial": torch.tensor(serial, dtype=torch.int),
        #     "solved_count": torch.tensor(solved_count, dtype=torch.float),
        #     "correct_before": torch.tensor(correct_before, dtype=torch.float),

        #     "wrong_before": torch.tensor(wrong_before, dtype=torch.float),
        #     "same_tag_solved_count": torch.tensor(same_tag_solved_count, dtype=torch.float),
        #     "same_tag_correct_before": torch.tensor(same_tag_correct_before, dtype=torch.float),
        #     "same_tag_wrong_before": torch.tensor(same_tag_wrong_before, dtype=torch.float),

        #     "item_correct_percent": torch.tensor(item_correct_percent, dtype=torch.float),
        #     "user_correct_percent": torch.tensor(user_correct_percent, dtype=torch.float),
        #     "current_correct_count": torch.tensor(current_correct_count, dtype=torch.float),
        #     "tag_group_one": torch.tensor(tag_group_one + 1, dtype=torch.int),

        #     "tag_group_two": torch.tensor(tag_group_two + 1, dtype=torch.int),
        #     "time_for_solve": torch.tensor(time_for_solve, dtype=torch.float),
        #     "guess_yn": torch.tensor(guess_yn, dtype=torch.int),
        #     "guess_yn_user": torch.tensor(guess_yn_user, dtype=torch.float),

        #     "guess_yn_test": torch.tensor(guess_yn_test, dtype=torch.float),
        #     "guess_yn_serial": torch.tensor(guess_yn_serial, dtype=torch.float),
        #     "guess_yn_assessment": torch.tensor(guess_yn_assessment, dtype=torch.float),
        #     "guess_yn_tag": torch.tensor(guess_yn_tag, dtype=torch.float),

        #     "guess_yn_day": torch.tensor(guess_yn_day, dtype=torch.float),
        #     "guess_yn_group_one": torch.tensor(guess_yn_group_one, dtype=torch.float),
        #     "guess_yn_group_two": torch.tensor(guess_yn_group_two, dtype=torch.float),
        #     "day_of_week": torch.tensor(day_of_week, dtype=torch.int),

        #     'zero': torch.tensor(zero, dtype=torch.float),
        #     'user_ability': torch.tensor(user_ability, dtype=torch.float),
        #     'day_correct_percent' : torch.tensor(day_correct_percent, dtype=torch.float),
        #     'correct_percent_group_one' : torch.tensor(correct_percent_group_one, dtype=torch.float),

        #     'correct_percent_group_two' : torch.tensor(correct_percent_group_two, dtype=torch.float),
        #     'correct_percent_serial' : torch.tensor(correct_percent_serial, dtype=torch.float),
        #     'duration_user' : torch.tensor(duration_user, dtype=torch.float),
        #     'user_mode_hour' : torch.tensor(user_mode_hour, dtype=torch.int32),

        #     'hour' : torch.tensor(hour, dtype=torch.int32),
        #     'year' : torch.tensor(year, dtype=torch.int32),
        #     'user_mode_year' : torch.tensor(user_mode_year, dtype=torch.float),
        #     'test_min_year' : torch.tensor(test_min_year, dtype=torch.float),

        #     'test_mode_year' : torch.tensor(test_mode_year, dtype=torch.float),
        #     'test_max_year' : torch.tensor(test_max_year, dtype=torch.float),
        #     'item_min_year' : torch.tensor(item_min_year, dtype=torch.float),
        #     'item_mode_year' : torch.tensor(item_mode_year, dtype=torch.float),

        #     'item_max_year' : torch.tensor(item_max_year, dtype=torch.float),
        #     'user_max_year' : torch.tensor(user_max_year, dtype=torch.float),
        #     'user_min_year' : torch.tensor(user_min_year, dtype=torch.float),
        #     'user_period_year' : torch.tensor(user_period_year, dtype=torch.float),

        #     'test_count' : torch.tensor(test_count, dtype=torch.float),
        #     'item_count' : torch.tensor(item_count, dtype=torch.float),
        #     'item_difficulty' : torch.tensor(item_difficulty, dtype=torch.float),
        #     'time_diff' : torch.tensor(time_diff, dtype=torch.float),

        #     'user_solve_count' : torch.tensor(user_solve_count, dtype=torch.float),

        embed = torch.cat(
            [
                embed_test,
                embed_tag,
                embed_question,
                embed_interaction,
                duration.unsqueeze(-1).float(),
                #startTime.unsqueeze(-1).float(),
                #elapsedTime.unsqueeze(-1).float(),
                embed_test_group_one,
                embed_test_group_two,
                embed_serial,
                solved_count.unsqueeze(-1).float(),
                correct_before.unsqueeze(-1).float(),
                wrong_before.unsqueeze(-1).float(),
                same_tag_solved_count.unsqueeze(-1).float(),
                same_tag_correct_before.unsqueeze(-1).float(),
                same_tag_wrong_before.unsqueeze(-1).float(),
                item_correct_percent.unsqueeze(-1).float(),
                user_correct_percent.unsqueeze(-1).float(),
                current_correct_count.unsqueeze(-1).float(),
                embed_tag_group_one,
                embed_tag_group_two,
                time_for_solve.unsqueeze(-1).float(),
                embed_guess,
                embed_guess_user,
                embed_guess_test,
                embed_guess_serial,
                embed_guess_assessment,
                embed_guess_tag,
                embed_guess_day,
                embed_guess_group_one,
                embed_guess_group_two,
                embed_day,
                zero.unsqueeze(-1).float(),
                user_ability.unsqueeze(-1).float(),
                day_correct_percent.unsqueeze(-1).float(),
                correct_percent_group_one.unsqueeze(-1).float(),
                correct_percent_group_two.unsqueeze(-1).float(),
                correct_percent_serial.unsqueeze(-1).float(),
                duration_user.unsqueeze(-1).float(),
                user_mode_hour.unsqueeze(-1).float(),
                hour.unsqueeze(-1).float(),
                year.unsqueeze(-1).float(),
                user_mode_year.unsqueeze(-1).float(),
                test_min_year.unsqueeze(-1).float(),
                test_mode_year.unsqueeze(-1).float(),
                test_max_year.unsqueeze(-1).float(),
                item_min_year.unsqueeze(-1).float(),
                item_mode_year.unsqueeze(-1).float(),
                item_max_year.unsqueeze(-1).float(),
                user_max_year.unsqueeze(-1).float(),
                user_min_year.unsqueeze(-1).float(),
                user_period_year.unsqueeze(-1).float(),
                test_count.unsqueeze(-1).float(),
                item_count.unsqueeze(-1).float(),
                item_difficulty.unsqueeze(-1).float(),
                time_diff.unsqueeze(-1).float(),
                user_solve_count.unsqueeze(-1).float(),
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
                zero,
                user_ability,
                day_correct_percent,
                user_mode_hour,
                hour,
                year,
                user_mode_year,
                test_min_year,
                test_mode_year,
                test_max_year,
                item_min_year,
                item_mode_year,
                item_max_year,
                user_max_year,
                user_min_year,
                user_period_year,
                test_count,
                item_count,
                time_diff,
                user_solve_count,
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
                                        zero=zero,
                                        user_ability=user_ability,
                                        day_correct_percent=day_correct_percent,
                                        user_mode_hour=user_mode_hour,
                                        hour=hour,
                                        year=year,
                                        user_mode_year=user_mode_year,
                                        test_min_year=test_min_year,
                                        test_mode_year=test_mode_year,
                                        test_max_year=test_max_year,
                                        item_min_year=item_min_year,
                                        item_mode_year=item_mode_year,
                                        item_max_year=item_max_year,
                                        user_max_year=user_max_year,
                                        user_min_year=user_min_year,
                                        user_period_year=user_period_year,
                                        test_count=test_count,
                                        item_count=item_count,
                                        time_diff=time_diff,
                                        user_solve_count=user_solve_count,
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
