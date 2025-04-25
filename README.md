# QuizBowl_QA
## What and Why Quizbowl
Quiz bowl is a competition that encompasses a broad range of topics. Two teams compete against each other simultaneously, competing to answer the question before there oponents. There are two key distinctions between quiz bowl and a typical quiz competition. Firstly, competitors have the ability to "buzz in" at any point during the question while it is being read by the moderator. This feature transforms the competition into a test of analytical thinking, rather than a recall task.
Another key difference lies in the format of the questions. Quiz bowl questions are typically longer and more detailed compared to those in a typical quiz competition. They consist of various somewhat unrelated clues that collectively lead to the answer. As the question progresses, the information provided becomes increasingly specific. This unique structure creates a competition where participants must strike a balance between confidence and speed. They need to accurately interpret the clues and make quick decisions on when to buzz in with their response.


Due to these unique factors, developing a model that can simulate quiz bowl competitors is very intriguing. In a traditional quiz competition, all participants can only buzz in once the entire question has been read aloud. This setup provides an unfair advantage to computers because they could immediately buzz in after confirming their guess, unaffected by physical limitations. Creating systems that preprocess and interpret queries in chunks could reduce perceived latency and increase responsiveness in general QA applications, not just quiz bowl.

## Example Question
### Text
    One family in this work flees France after rescuing an "unfortunate Muhammadan" who is arrested for his
    religion the day his daughter Safie arrives from Constantinople. In addition to the De Laceys, another 
    character in this work educated himself by reading about the Northwest Passage, and begins the novel by 
    writing a series of letters to Margaret. Justine Moritz is falsely accused of the murder of the title 
    character's brother William in this novel, which causes the title character to travel to England with
    Clerval to satisfy the demands of the antagonist for a female companion. The protagonist is rescued by 
    Robert Walton in the Arctic and hails from Lake Geneva. For 10 points, identify this novel subtitled 
    The Modern Prometheus, a Mary Shelley work about creating a certain monster.
### Answer
    Frankenstein
    
## Methodology 
### Method 1 
The first method is a information retrieval model that relies the Pyserini library to retrieve the top document with the highest similarity to the query. The model then extracts their associated answers. There are two implementation options available. The retrival model will use a pre-made index assocuiated with a wikipedia document corpus. Once the most similar document is retrieved the first named entity is extracted and used for the answer. 

### Method 2 
The second method utilized a fine-tuned question answering model using the same retrieval functions and BERT. In this approach, Pyserini was used to retrieve the most closely related document. BERT requires both context and the actual query, Pyserini assists by finding a document that potentially contains the answer, this document will be truncated and used for the model's context. 


## Notes 
if you want to test the program you can choose to run the test.py file like this: 

        python3 qbam.py 
        
        
If you run the code, it will generate the guess using BERT and a pre-made Pyserini index. It's important to note that this index is quite large, exceeding 10GB in size. Therefore, be prepared for the storage requirements and potential download time.

