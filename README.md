# QuizBowl_QA
### CMSC470 Final project, data sourced from the class repository.
---
## What and Why Quizbowl
Quiz bowl is a competitive event that encompasses a diverse range of topics. Typically, two teams compete against each other simultaneously. There are two key distinctions between quiz bowl and a typical quiz competition. Firstly, competitors have the ability to "buzz in" at any point during the question while it is being read by the moderator. This feature transforms the competition into a test of analytical thinking and overall knowledge rather than a passive recall exercise.
Another key difference lies in the format of the questions. Quiz bowl questions are typically longer and more detailed compared to those in a typical quiz competition. They consist of various seemingly unrelated clues that collectively lead to the answer. As the question progresses, the information provided becomes increasingly specific. This unique structure creates a nuanced competition where participants must strike a balance between confidence and speed. They need to accurately interpret the clues and make quick decisions on when to buzz in with their response.

Due to these unique factors, developing a model that can emulate quiz bowl competitors becomes even more intriguing. In a traditional quiz competition, all participants can only buzz in once the entire question has been read aloud. This setup would provide an unfair advantage to computers, as they could immediately buzz in after confirming their guess, unaffected by the limitations of motor control. Furthermore, computers would be able to swiftly retrieve similar questions and consistently make accurate guesses. However, the format of quiz bowl levels the playing field, allowing humans to compete against computers on a more equal footing. This format provides a better representation of the model's "intelligence".


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

This project employs two methods to generate guesses. The first method is a straightforward information retrieval model that relies on the top n documents with the highest similarity to the query. The model selects these documents and extracts the corresponding questions and their associated answers. There are two implementation options available. The first option is to create a BM25 index from scratch by building an index using quiz bowl questions. This approach ensures that the index is specifically tailored for quiz bowl content. The second option is to download a pre-made index; however, it is recommended to create a custom index to better handle the unique characteristics of quiz bowl questions. Other existing indexes may struggle with longer queries and could retrieve loosely related documents, potentially impacting the accuracy of the generated guesses.

The second method utilized a pretrained question answering model using Pyserini and BERT. In this approach, Pyserini was employed to retrieve the most closely related document. BERT requires both context and the actual query to work effectively. Pyserini assists by finding a document that potentially contains the answer to the question. However, there are certain constraints to consider. Both the question and the context must not exceed 512 tokens, where each token represents a word. This limitation necessitates truncating the document while ensuring that it still contains the answer.

To address these challenges, the "wikipedia-kilt-doc" index was utilized. This index was selected because the document titles often corresponded to the answer, and the titles were generally located at the beginning of the document. By leveraging this index, the model could better accommodate the token limitations while increasing the chances of capturing the relevant answer within the truncated document.


## Notes 
if you want to test the program you can choose to run the test.py file like this: 

        python3 test.py --data = evaluation.json
        
        
If you run the code without making any changes, it will generate the guess using BERT and a pre-made Pyserini index. It's important to note that this index is quite large, exceeding 10GB in size. Therefore, be prepared for the storage requirements and potential download time.

If you wish to change the default index, you can navigate to the "Context_Generator.py" file and modify line 36 to specify the desired index.

Alternatively, if you prefer to use the pre-made index of quiz bowl questions provided in this repository, you can modify the "Guesser.BertGuess" flag in the "qbmodel" file to set it to "false."

Finally, if you opt for the pure retrieval method, you can uncomment the corresponding sections labeled "pure retrieval" in the "qbmodel" file. This will enable the pure retrieval approach for generating guesses.

Please note that making these changes requires familiarity with the code structure and implementation details.
