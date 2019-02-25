# sentiment_classifier.py

import argparse
import sys
import time
from models import *
from sentiment_data import *

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


# Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
# are provded for convenience.
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='NB', help='model to run (TRIVIAL, NB, or PERCEPTRON)')
    parser.add_argument('--feats', type=str, default='UNIGRAM', help='feats to use (UNIGRAM, BIGRAM, or BETTER)')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


# Evaluates a given classifier on the given examples
def evaluate(classifier, exs):
    print_evaluation([ex.label for ex in exs], [classifier.predict(ex) for ex in exs])


# Prints accuracy comparing golds and predictions, each of which is a sequence of 0/1 labels.
# Prints accuracy, and precision/recall/F1 of the positive class, which can sometimes be informative if either
# the golds or predictions are highly biased.
def print_evaluation(golds, predictions):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)
    test_exs = read_sentiment_examples(args.blind_test_path)
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples")
    
    # Train and evaluate
    start_time = time.time()
    model = train_model(args, train_exs)
    print("=====Train Accuracy=====")
    evaluate(model, train_exs)
    print("=====Dev Accuracy=====")
    evaluate(model, dev_exs)
    #print("=====Test Accuracy=====")
    #evaluate(model, test_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))

    # Write the test set output
    if args.run_on_test:
        test_exs_predicted = [SentimentExample(ex.words, model.predict(ex)) for ex in test_exs]
        write_sentiment_examples(test_exs_predicted, args.test_output_path)