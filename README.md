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
To achieve this I chose to use Pyserini and Bert, Pysernin
