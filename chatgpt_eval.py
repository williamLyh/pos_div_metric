from openai_utils import call_openai_chat_completion
from jinja2 import Environment
from textwrap import dedent
import json
import ast
from tqdm import tqdm
from datasets import load_dataset

#    2. Consider the flow of ideas and the ordering of sentences. A highly coherent article should have a clear and smooth progression of thoughts, with well-established links between concepts. 
#     3. Take into account how effectively the introduction, body, and conclusion contribute to the overall cohesion of the content. Provide specific examples or areas for improvement if necessary.
 
def get_chatgpt_eval_news(dataset, engine='gpt-3.5-turbo'):
    evaluation_prompt = dedent("""\
        Pretend you are a human reader. Please evaluate the coherence of the two given news articles.
        Guideline:
        1. Rate on a scale of 1 to 10, where 1 represents very low coherence, and 10 indicates very high coherence.
        2. Consider the flow of ideas and the ordering of sentences. A highly coherent article should have a better sentence ordering. 
        3. Must return ratings in JSON format only: {"score1": [your rating for version 1], "score2": [your rating for version 2]}

        News headline:
        {{ headline }}

        News version 1:
        {{ version1 }}

        News version 2:
        {{ version2 }}

        Rating:\
    """)
    environment = Environment()
    evaluation_prompt = environment.from_string(evaluation_prompt)

    gpt_scores = []
    gpt_scores_unshuffled = []
    for id in tqdm(range(dataset.shape[0])):
        try:
            prompt = evaluation_prompt.render(
                headline=dataset['input'][id],
                version1=dataset['output_shuffled_1'][id],
                version2=dataset['output_shuffled_2'][id],
                # version3=dataset['output_shuffled_2'][id]
            )
            random_indice = dataset['random_indice'][id]
            llm_output = call_openai_chat_completion(prompt, model=engine)
            if llm_output:
                corrected_score = [list(llm_output.values())[i] for i in random_indice]
                gpt_scores.append(list(llm_output.values()))
                gpt_scores_unshuffled.append(corrected_score)
            else:
                gpt_scores.append(None)
                gpt_scores_unshuffled.append(None)
        except Exception as e:
            print('Too long? ', e)
            pass
    dataset = dataset.add_column("gpt_scores", gpt_scores)
    dataset = dataset.add_column("gpt_scores_unshuffled", gpt_scores_unshuffled)
    return dataset


def get_chatgpt_eval_lfqa(dataset, engine='gpt-3.5-turbo'):

    evaluation_prompt = dedent("""\
        Pretend you are a human reader. Please evaluate the coherence of the two given answer paragraphs.
        Coherence guidelines:
        1. Evaluate how well the sentences transition from one to another. A fluent text should have seamless connections between sentences.
        2. Evaluate how well the sentences are organized and the ideas are conveyed. A coherent answer paragraph should have a clear and precise structure.
        General guidelines:
        1. Rate the coherence of the text from 1 to 10, where 1 is the lowest and 10 is the highest.
        2. Utilize the entire rating scale, from the lowest to the highest score, to provide nuanced feedback.
        3. Must return ratings in JSON format only: {"score1": [your rating for version 1], "score2": [your rating for version 2]}

        Question:
        {{ headline }}

        Answer version 1:
        {{ version1 }}

        Answer version 2:
        {{ version2 }}

        Rating:\
    """)

    environment = Environment()
    evaluation_prompt = environment.from_string(evaluation_prompt)

    gpt_scores = []
    gpt_scores_unshuffled = []
    for id in tqdm(range(dataset.shape[0])):
        try:
            prompt = evaluation_prompt.render(
                headline=dataset['input'][id],
                version1=dataset['output_shuffled_1'][id],
                version2=dataset['output_shuffled_2'][id],
                # version3=dataset['output_shuffled_2'][id]
            )
            random_indice = dataset['random_indice'][id]
            llm_output = call_openai_chat_completion(prompt, model=engine)
            if llm_output:
                corrected_score = [list(llm_output.values())[i] for i in random_indice]
                gpt_scores.append(list(llm_output.values()))
                gpt_scores_unshuffled.append(corrected_score)
            else:
                gpt_scores.append(None)
                gpt_scores_unshuffled.append(None)
        except Exception as e:
            print('Too long? ', e)
            pass
    dataset = dataset.add_column("gpt_scores", gpt_scores)
    dataset = dataset.add_column("gpt_scores_unshuffled", gpt_scores_unshuffled)
    return dataset


def get_chatgpt_eval_recipe(dataset, engine='gpt-3.5-turbo'):

    evaluation_prompt = dedent("""\
        Please pretend you are a human reader. Read the tree versions of the recipes below for a given dish title and rate their coherence scores following the guideline.
        Coherence guidelines:
        1. Evaluate how well the sentences transition from one to another. A fluent text should have seamless connections between sentences.
        2. Evaluate how well the sentences are organized and the ideas are conveyed. A coherent text should have a clear and precise structure.
        General guidelines:
        1. Rate the coherence of the text from 1 to 10, where 1 is the lowest and 10 is the highest.
        2. Utilize the entire rating scale, from the lowest to the highest score, to provide nuanced feedback.
        3. Please return in JSON format. For example, {"score1": 1, "score2": 2, "score3": 3}.

        Dish title:
        {{ headline }}

        Recipe version 1:
        {{ version1 }}

        Recipe version 2:
        {{ version2 }}

        Rating:\
    """)

    environment = Environment()
    evaluation_prompt = environment.from_string(evaluation_prompt)

    gpt_scores = []
    gpt_scores_unshuffled = []
    for id in tqdm(range(dataset.shape[0])):
        try:
            prompt = evaluation_prompt.render(
                headline=dataset['input'][id],
                version1=dataset['output_shuffled_1'][id],
                version2=dataset['output_shuffled_2'][id],
                # version3=dataset['output_shuffled_2'][id]
            )
            random_indice = dataset['random_indice'][id]
            llm_output = call_openai_chat_completion(prompt, model=engine)
            if llm_output:
                corrected_score = [list(llm_output.values())[i] for i in random_indice]
                gpt_scores.append(list(llm_output.values()))
                gpt_scores_unshuffled.append(corrected_score)
            else:
                gpt_scores.append(None)
                gpt_scores_unshuffled.append(None)
        except Exception as e:
            print('Too long? ', e)
            pass
    dataset = dataset.add_column("gpt_scores", gpt_scores)
    dataset = dataset.add_column("gpt_scores_unshuffled", gpt_scores_unshuffled)
    return dataset