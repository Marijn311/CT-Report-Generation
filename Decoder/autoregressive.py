from config import *
from dataset import *
import evaluate #pip3 install evaluate

    
class autoregressive():
    """Class to perform autoregressive inference with a PyTorch decoder model.
    
    Transformer models use parallelization and teacher forcing during training (and validation). 
    This means that every word in a sequence is predicted at once, using all previous words in the ground truth as input/context for the model.  
    This way, errors do not accumilate in the context and training goes smoother.
    However, during autoregressive inference no teacher forcing should be applied.
    Adapting the models to NOT use teacher forcing is no trivial task.
     
    The solution to this problem is this class.
    This class loads a pre-trained model. Instead of passing the reference reports at t=0, the input is set to be 'sos'.
    From the generated report, only the next word is extracted (eventhough it return all the predicted words up until it reaches target length)
    The predicted next word is added to the new input so the input becomes "sos + prediction 1" at t=1 and then extract the predicted word at position 3.
    This pattern continues until there are max length -1 predictions.
    
    This class also prints the available context, the ground truth next word, and the predicted next word for each prediction.
    Latsly, this class calculates the BLEU, METEOR, and ROUGE scores for the entire dataset.
    
    Attributes:
        model (torch.nn.Module): The pre-trained PyTorch decoder model.
        data_module: The data module used to run inference on.
        token_to_word (dict): A dictionary mapping token indices to word strings.
        bleu: The BLEU metric evaluator.
        meteor: The METEOR metric evaluator.
        rouge: The ROUGE metric evaluator.
        metrics (pd.DataFrame): A dataframe to store the evaluation metrics.
        
    Methods:
        print_predictions(inputs, reports, preds, i): Print the available context, the ground truth next word, and the predicted next word for every prediction.
        evaluate_predictions(inputs, reports, preds): Calculate the BLEU, METEOR, and ROUGE scores for the predicted sentences.
        autoregressive_inference(): Perform autoregressive inference on the entire dataset.
    """
    
    def __init__(self, model, data_module):
        super().__init__()
        
        # Load the token to word dictionary
        self.token_to_word = {v: k for k, v in VOCAB_DICT.items()}
        
        # Initialize NLP metrics and a dataframe to store the metrics    
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')
        self.metrics = pd.DataFrame(columns=['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rouge2'])
        
        # Set up the model and data_module/data_loader
        self.model=model.to('cuda').eval()
        data_module.setup(stage=STAGE)
        if AUTOREGRESSIVE_DATASET == 'train':
            self.inference_dataloader = data_module.train_dataloader()
        if AUTOREGRESSIVE_DATASET == 'val':
            self.inference_dataloader = data_module.val_dataloader()
        if AUTOREGRESSIVE_DATASET == 'test':
            self.inference_dataloader = data_module.test_dataloader()
    
            
    def print_predictions(self, inputs, reports, preds, i):
        """Print the available context, the ground truth next word, and 
        the predicted next word for every prediction."""
       
        # Only print 1 sample per batch
        input=inputs[0,:]
        report=reports[0,:]
        pred=preds[0,:]
        
        # Get the context, ground truth next word, and predicted next word at i
        context = input[:i+1] 
        gt_next_word = report[i+1] 
        pred_next_word = pred[i]
        gt_next_word = gt_next_word.unsqueeze(0)
        pred_next_word = pred_next_word.unsqueeze(0)
        
        # Convert the token indices to word strings
        untokenised_context = [self.token_to_word[word.item()] for word in context]
        untokenised_gt_next_word = self.token_to_word[gt_next_word.item()]
        untokenised_pred_next_word = self.token_to_word[pred_next_word.item()]
         
        print(f"\nAvailable context: {untokenised_context}")
        print(f"GT next word: {untokenised_gt_next_word}")
        print(f"Predicted next word: {untokenised_pred_next_word}")
        



    def evaluate_predictions(self, inputs, reports):
        """Print the predicted sentences against the ground truth.
        Calculate the BLEU, METEOR, and ROUGE scores for the predicted sentences."""
        
        # Evaluation is done seperately for each sentence in the batch
        for sample in range(BATCH_SIZE):
            # Get the generated and ground truth reports. Convert to numpy arrays
            gen_report = inputs[sample, :].clone().cpu().numpy()
            gt_report = reports[sample, :].clone().cpu().numpy()
            
            # Convert the token indices to word strings. Remove the sos/eos/pad tokens
            gen_report = [self.token_to_word[idx] for idx in gen_report]
            clean_gen_report = []
            for i, string in enumerate(gen_report): # Remove all the tokens after and including the first eos token
                if string != 'eos':
                    clean_gen_report.append(string)
                else:
                    break
            gen_report = clean_gen_report
            gen_report = [x for x in gen_report if x != '<PAD>' and x != '[PAD]' and x != 'sos' and x != 'eos' and x != '[CLS]' and x != '[SEP]']    #remove any occurances of pad 
            gen_report = ' '.join(gen_report)
            #fix for bioclinical bert the sup word tokens need to be joined into normal tokens again before we pass them top blue en meter etc
            gt_report = [self.token_to_word[idx] for idx in gt_report]
            gt_report = [x for x in gt_report if x != '<PAD>' and x != '[PAD]' and x != 'sos' and x != 'eos' and x != '[CLS]' and x != '[SEP]']    #remove any occurances of pad or sos os eos
            gt_report = ' '.join(gt_report)
            
            print(f"\nGround truth: {gt_report}") 
            print(f"\nPrediction: {gen_report}")
            
            # Calculate NLP metrics and store them in a dataframe
            bleu_score = self.bleu.compute(predictions=[gen_report], references=[gt_report])
            meteor_score = self.meteor.compute(predictions=[gen_report], references=[gt_report])
            rouge_score = self.rouge.compute(predictions=[gen_report], references=[gt_report])
            
            print(f"\nBLEU SCORE: {bleu_score}") 
            print(f"\nMETEOR SCORE: {round(meteor_score['meteor'], 2)}")
            print(f"\nROUGE SCORE: {rouge_score}")
                
            new_metrics = pd.DataFrame({'bleu': [bleu_score['bleu']], 'bleu1': [bleu_score['precisions'][0]], 'bleu2': [bleu_score['precisions'][1]], 'bleu3': [bleu_score['precisions'][2]], 'bleu4': [bleu_score['precisions'][3]], 'meteor': [meteor_score['meteor']], 'rouge1': [rouge_score['rouge1']], 'rouge2': [rouge_score['rouge2']]})
            self.metrics = pd.concat([self.metrics, new_metrics], ignore_index=True)
            
            
    def autoregressive_inference(self):
        """Perform autoregressive inference on the entire dataset. This function calls on print_predictions and evaluate_predictions."""
        
        # Loop over all batches
        for batch in self.inference_dataloader: 
            
            # Load a batch of encoded images, reference reports and initialize the input tensor with SOS token
            images, reports = batch
            images = images.to('cuda')
            reports = reports.to('cuda')
            inputs = torch.full((BATCH_SIZE, MAX_LENGTH), SOS_IDX).to('cuda')
            
            # A loop to predict the next word in the report. The loop stops when the max length-1 is reached. (Not when the eos token predicted because the predictions are done in batches) 
            for i in range(MAX_LENGTH-1):
                # Forward pass the input tensor through the model to get predictions
                with torch.no_grad():
                    logits = self.model.forward(images, inputs)
                preds = torch.argmax(logits, dim=-1)
                
                self.print_predictions(inputs, reports, preds, i)

                # Update the input tensor with the predicted next word
                inputs[:, i+1] = preds[:, i]
                
            self.evaluate_predictions(inputs, reports)

        # Print the average BLEU, METEOR, and ROUGE scores
        avg_metrics = self.metrics.mean(axis=0)
        print(f"\nAverage BLEU: {avg_metrics['bleu']}") 
        print(f"\nAverage BLEU1: {avg_metrics['bleu1']}")
        print(f"\nAverage BLEU2: {avg_metrics['bleu2']}")
        print(f"\nAverage BLEU3: {avg_metrics['bleu3']}")
        print(f"\nAverage BLEU4: {avg_metrics['bleu4']}")
        print(f"\nAverage METEOR: {avg_metrics['meteor']}")
        print(f"\nAverage ROUGE1: {avg_metrics['rouge1']}")
        print(f"\nAverage ROUGE2: {avg_metrics['rouge2']}")
    

   









