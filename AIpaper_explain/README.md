# Explained - *An image is worth 16x16 words: transformers for image recognition at scale*

## Motivations
I wanted to talk about this paper because I found it so clever and I was so exciting when I read it. 

First of all, it talks about pretraining on 303M high resolution images ! The fact that we can deal with a such amont of high resolution data is impressive to me (ok, we need the computational power of Google but still). 

Then, this paper trigger my engineer nature: in product engineering, starting a new project comes with high expenses especially in aeronautic where I come frome. To innovate on a new architecture one should reconsider the whole process, from research offices to assembly lines. So, the first question is *what do we have "on shelves" that works, and that could fit with this new project ?*. I see this paper exactly like that, it's why I found it so clever. One knew that self attention models was a game changer in NLP problems so the authors came with this idea *"could we reuse what is actually working well on an other scope of problem ?"*.