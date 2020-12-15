"""Scorer"""
import nltk

class BleuScorer(object):
    """Blue scorer class"""
    def __init__(self):
        self.results = []
        self.results_meteor = []
        self.score = 0
        self.meteor_score = 0
        self.instances = 0
        self.meteor_instances = 0

    def example_score(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    
    def example_score_meteor(self, reference, hypothesis):
        """Calculate blue score for one example"""
        return nltk.translate.meteor_score.single_meteor_score(reference,hypothesis)

    def data_score(self, data, predictor):
        """Score complete list of data"""
        results_prelim = []
        for example in data:
            #i = 1
            src = [t.lower() for t in example.src]
            reference = [t.lower() for t in example.trg]
            # loop through example.src and calculate all hypothesis(max. 8) 
            #and calculate blue score average of all hypothesis
            hypothesis = predictor.predict(example.src)
            blue_score = self.example_score(reference, hypothesis)
            meteor_score = self.example_score_meteor(' '.join(reference), ' '.join(hypothesis))
            #print('Blue Score: ',blue_score)
            results_prelim.append({
                'question': '"' + str(src) + '"',
                'reference': reference,
                'hypothesis': hypothesis,
                'blue_score': blue_score,
                'meteor_score': meteor_score
            })
        #print('List length before aggregation',len(results_prelim))

        results = [max((v for v in results_prelim if v['question'] == x), key=lambda y:y['blue_score']) for x in set(v['question'] for v in results_prelim)] 

        with open('result_output.txt', 'w') as f:
            for elem in results:
                f.write("%s\n" % elem)
                self.results.append(elem)
                self.score += elem['blue_score']
                self.meteor_score += elem['meteor_score']
                self.instances += 1
        return self.score / self.instances, self.meteor_score / self.instances

    def average_score(self):
        """Return bleu average score"""
        return self.score / self.instances
    
    def data_meteor_score(self, data, predictor):
        """Score complete list of data"""
        results_prelim = []
        for example in data:
            src = [t.lower() for t in example.src]
            reference = [t.lower() for t in example.trg]
            hypothesis = predictor.predict(example.src)
            meteor_score = self.example_score_meteor(' '.join(reference), ' '.join(hypothesis))
            results_prelim.append({
                'question': '"' + str(src) + '"',
                'reference': reference,
                'hypothesis': hypothesis,
                'meteor_score': meteor_score
            })
        results_meteor = [max((v for v in results_prelim if v['question'] == x), key=lambda y:y['meteor_score']) for x in set(v['question'] for v in results_prelim)] 

        with open('result_meteor_output.txt', 'w') as f:
            for elem in results_meteor:
                f.write("%s\n" % elem)
                self.results_meteor.append(elem)
                self.meteor_score += elem['meteor_score']
                self.meteor_instances += 1
        return self.meteor_score/self.meteor_instances
    
    def average_meteor_score(self):
        """Return meteor average score"""
        return self.meteor_score/self.instances

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.results_meteor = []
        self.score = 0
        self.meteor_score = 0
        self.instances = 0
        self.meteor_instances = 0
