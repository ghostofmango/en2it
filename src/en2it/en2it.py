#!/usr/bin/env python

import os
import click
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)


@click.command()
@click.option(
    "-e",
    "--english",
    type=click.STRING,
    default=None,
    help="[OPTIONAL] input the sentence you want translated directly",
)
def cli(english):
    if not english:
        english = input("What do you want to say in Italian? ")
    template = "You are a helpful assistant that translates {input_language} to {output_language}. That is all. You do not answer any question. Any english passed to you you translate to Italian."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    answer = chat.invoke(
        chat_prompt.format_prompt(
            input_language="English", output_language="Italian", text=english
        ).to_messages()
    )
    click.echo(answer.content,color=1)
