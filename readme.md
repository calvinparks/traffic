<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 0.401 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β31
* Sun Nov 21 2021 15:42:24 GMT-0800 (PST)
* Source doc: Traffic Signs Read.Me
----->


**What did you try?**

**(See below for more details)**

**Different number of Convolutional layers and settings**

**Different number of BatchNormalization layers**

**Different number of Dense layers and settings**

**Different Dropout percentages**

**Different Compiler options **

**What worked well?**

**(See below for more details)**

**Performing lot’s of research, reading tensorflow and keras  documentation.**

**Reading what Lenet is.**

**Researching detail how filters are actually created**

**Using 3 convolutional layers and only 1 pooling layer at the end of them. **

**Using 1 Batch Normalization layer between Dense layers**

**What didn’t work well?**

**(See below for more details)**

**Pooling between Convolutional networks**

**Many settings I experimented with during Trial and Error.**

**More than 1 Batch Normalization layer  that causes overfitting**

**More than 3  convolutional layers that causes overfitting**

**What did you notice?**

**Finally,  I learned that when trying to figure out the best CCN code, it will be the code that causes the training  epochs to start out with the lowest accuracy values which smoothly increase with each epoch to be the final training  epoch accuracy around 97%.**



**Detailed Explanation of my process and solution**

Project Goal: Write an AI to identify which traffic sign appears in a photograph. Uses OpenCV and TensorFlow and the  "German Traffic Sign Recognition benchmark" Dataset of 43 different kinds of road signs

I wrote the code for creating the model within the get_model() and load_data() functions

First I looked at the openCV documentation to understand how to use it to read in the image data and make it return data formatted as a numpy array which was required by the specifications. I also read it to understand the various interpolation settings; initially I set it to INTER_AREA after I created the rest of the code for the project; I went back and systematically changed the interpolation to all the other possible values. Then I ran a test for each and notated the resulting test accuracy.  Then I chose INTER_LANCZOS4 as my final choice because it was the one with the highest average accuracy. 

After reading the openCV docs, I read the The official Tensorflow documentation and Keras documentation to help me understand the typical commands used when designing a CNN

After that I was still uncertain about how the various filters were created under the hood and what to actually see. So I viewed many youtube videos for several hours until I felt confident I knew how they work.


# Before I began to create the function “get_model”  Python code, I reviewed videos and summaries of the legendary 1998 **Lenet **paper to understand the original solution for solving this category of a problem.

Then I finally felt confident to begin coding the “get_model” function. Even though today’s technology is far more advanced than the 1998 when the Lenet paper was published,  My initial code framework roughly followed the Lenet pattern. However the final code is significantly different because I wanted to find a different pattern of layers that worked with high accuracy.

Since the traffic signs have 3 color channels I wanted to use three Conv2d  layers. The layers  increasing filter size would be  2x2 for the 1st layer.  4x4 for the 2nd layer and 8x8 for the 3rd. They should be finding the primitive elements to more complex structures.

I decided to only use 1 pooling layer after the final Conv2d layer because I reduce image detail loss during the 3 Conv2d layers processing.

Then I flattened the layers before passing my dense hidden layer processed the images. 

First I set the dense layer count to 128. But that failed by giving me quickly accelerating accuracy for each Epoch. Using trial and error I experimented with the hidden layer dimensionality and settled on 64 because it consistently  supported  lowest Epoch-1 accuracy which increased modestly increased for each subsequent Epoch  which ended on Epoc-10 above 95% accuracy

My model result accuracy was not stable between test runs.  After doing extensive research I discovered it might be because the data scales changed over epochs so I found out that BatchNormalization will scale all the intermediate data between layers is uniformly scaled between 0 and 1

Then I added a Dropout layer set at 50%.  Later after everything else was working I experimented more with the Dropout layer setting and found 54% increased the final accuracy by about 5% before passing it to the final Dense layer 

The final Dense layer is the output layer which outputs units of percentages for all all possible categories.

After I completed all the layers. I read the Keras documentation to understand the various compiler options.  Then  I tried setting the compiler options to a variety of settings and the following  choices performed best.  

I thought the metrics=“accuracy” would be the best option but it was causing overfitting so I chose to use  the "CategoricalAccuracy"] instead.

 

model.compile(

        optimizer="nadam",

        loss="categorical_crossentropy",

        metrics=["CategoricalAccuracy"]

    )

 

**ATTENTION. Since my computer uses an AMD video card, TensorFlow cannot access the GPU so it defaults to using the slower CPU.  **

**As a result of this, when running my code you will see 2 warning messages referring to “_Could not load dynamic library 'cudart64_110.dll'”  _**

**_Both warnings <span style="text-decoration:underline;">can be ignored</span>._**

**This screen video capture is sped up because the CPU is much slower than the GPU so the program runtime is actually 00:04:49  And the project specifications said this video must be under 00:05:00 minutes.**


HERE IS A YOUTUBE LINK THAT SHOWS THIS PROJECT WORKING [PROJECT VIDEO](https://www.youtube.com/watch?v=yuqv5lfRNl4).

My favorite search engine is 