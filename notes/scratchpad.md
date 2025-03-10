# Multimodal Medical Image Chat with Reasoning

## Project Idea

- Pocket Dermatologist
  - Small Llama 3.2 (3B or 8B) fine-tuned to demonstrate Reasoning abilities
  - Obtain image dataset of skin conditions
    ([Dermnet](https://dermnetnz.org/dermatology-image-dataset) seems to be worth
    exploring)
  - Figure out how to put in scan images into the model. There might be
    fine-tuned CLIP or other models that can do this.
  - Train model to predict skin conditions, and provide conservative,
    actionable feedback based on pictures taken

## TODOs

1. Explore HF course on Reasoning Models
2. Read up on [FineMed](https://github.com/hongzhouyu/FineMed), a repo about
   fine-tuning llama 3.1 8B to have medical reasoning abilities
3. Read the [SkinCAP paper](http://arxiv.org/abs/2405.18004)
4. Read the [FineMedLM-o1 paper](http://arxiv.org/abs/2501.09213)
5. Check out [SkinGPT-4](https://github.com/JoshuaChou2018/SkinGPT-4)
6. Make this week about writing a kickass Medium article

## LLM Post-Training

- Research community is now shifting focus to post-training techniques to
  achieve breakthroughs
- Check this ![graphic](/002%20󰸭%20%20LLM_post_training.png)
- Three main strats are dominating
  1. Fine-Tuning
  2. Reinforcement Learning
  3. Test-Time Scaling

## Techniques to Try

- Take FineMed Dataset and really understand it
- Try to fine-tune a Llama 3B with Fine-Med, doing multiple passes with
  increasingly complex medical fields
- This should culminate in Dermatological data
- Bes' suggestion
  - Take "high quality" outputs from a larger model
  - Use them as inputs to a smaller model, asking it to paraphrase
  - These paraphrased samples become the positive samples
  - Pass in the context raw into the model and ask it to generate responses
  - This will likely be less sohpisticated and will be the negative sampeles
  - Try a contrastive learning objective using Policy Optimzation based methods.
