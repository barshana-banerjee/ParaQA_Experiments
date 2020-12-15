"""Scorer"""
import nltk

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.score = 0
        self.instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def data_score(self, data, predictor):
        """Score complete list of data"""
        for example in data:
            #print('example target: ',example.trg)
            #for t in example.trg: 
            #    print('example target: ', t)
            #print('---------------- target --------------------')
            reference = [t.lower() for t in example.trg]
            #print(reference)
            #print('---------------- source --------------------')
            #source_ref = [t.lower() for t in example.src]
            #print(example.src)
            # loop through example.trg and calculate all hypothesis(max. 8) 
            #and calculate blue score average of all hypothesis
            #print('---------------- hypothesis --------------------')
            hypothesis = predictor.predict(example.src)
            #print(hypothesis)
            #num_ans = 0
            #blue_score = 0
            blue_score = self.example_score(reference, hypothesis)
            '''for ans in reference:
                if len(ans) > 1:
                    reference_ans = [t.lower() for t in ans.split(' ')]
                    blue_score = blue_score + self.example_score(reference_ans, hypothesis)
                    print('Answer: ',reference_ans)
                    print('Single Blue score: ',blue_score)
                    num_ans = num_ans + 1
                    '''
            #print('Sum blue_score: ',blue_score)
            #print('Total ans: ',num_ans)
            #blue_score = blue_score / num_ans
            #print('Final blue_score: ',blue_score) 
            self.results.append({
                'reference': reference,
                'hypothesis': hypothesis,
                'blue_score': blue_score
            })
            self.score += blue_score
            self.instances += 1

        return self.score / self.instances

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.score = 0
        self.instances = 0
