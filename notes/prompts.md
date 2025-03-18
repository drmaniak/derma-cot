# Medical Question Generation Prompt

Given our SCIN & SkinCAP datasets have now had AI generated labels, we need to generate medically relevant questions that mimic the style of a patient asking the doctor

'''
You need to generate two questions based on the given text content. These
questions must mimic the style of a patient who goes to visit a dermatologist
with a skin condition, but avoid relying on specific case details or diagnosis
from the text.

Follow these requirements:

Requirements:

1. Make sure the questions mimic a typical patient asking a doctor about
   their ailment, and as such must use simple language, avoiding complex
   medical jargon.
2. Ensure the two questions are as diverse as possible, avoiding homogeneity.
3. Ensure the questions include all the information needed for the answers.
   If necessary, add introductory information to the questions.
4. Avoid repetitive or overly simplistic questions, ensuring diversity and depth.
5. The questions must be self-contained and should not require the provided
   text as background to be understood.

Please rewrite the following text into related questions, and strictly follow
the format below for output:
{ "question1": "Generated first question", "question2": "Generated second
question" }
'''
