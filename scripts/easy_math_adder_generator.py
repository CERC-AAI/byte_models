import argparse
import json
import random
random.seed(1337)

from pathlib import Path

class AdditionDatasetGenerator():
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.

    Args:
        ndigit: Number of digits for each number that will be summed. Default is 2.
        test_ratio: Proportion of the total possible ndigit summation problems that will be in test. Default is 0.2
        reverse_answer: Reverse the answer number (since tokens are predicted left-to-right), this may ease the difficulty for the model
    """

    def __init__(self, ndigit=2, test_ratio=0.2, reverse_answer=True):

        # split up all addition problems into either training data or test data
        self.ndigit = ndigit
        self.test_ratio = test_ratio
        self.reverse_answer = reverse_answer
        # assert ndigit <= 4, "For ndigit > 4, it'll take a long time"
        assert test_ratio < 1.0, "Test ratio has to be less than 1"

        self.total_num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        print(f"Total possible problems for {ndigit}-digit addition: {self.total_num}")

        self.numbers = [i for i in range(10**ndigit)]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def generate_problems(self, save_dir):
        num_train = int(self.total_num * (1-self.test_ratio))
        num_test = int(self.total_num - num_train)

        train_file = Path(save_dir) / "train.txt"
        test_file = Path(save_dir) / "test.txt"
        print(f"Test ratio: {self.test_ratio}. Saving train file at {train_file} and test file at {test_file}")
        print(f"Number of training examples: {num_train}. Number of test examples: {num_test}")
        a_numbers = random.sample(self.numbers, len(self.numbers))
        b_numbers = random.sample(self.numbers, len(self.numbers))

        with open(train_file, 'w') as train_fp, open(test_file, 'w') as test_fp:
            idx = 0
            for a in a_numbers:
                for b in b_numbers:
                    if idx % 1000000 == 0:
                        print(f"Processing index {idx}")            

                    # calculate the "label" of the addition problem a + b
                    c = a + b
                    # encode the digits of a, b, c into strings
                    astr = f'%0{self.ndigit}d' % a
                    bstr = f'%0{self.ndigit}d' % b
                    cstr = (f'%0{self.ndigit+1}d' % c)[::-1] if self.reverse_answer else (f'%0{self.ndigit+1}d' % c)
                    render = astr + bstr + cstr

                    # Save to train file
                    if idx < num_train:
                        train_fp.write(render + '\n')

                    # Save to test file
                    else:
                        test_fp.write(render + '\n')

                    idx += 1

def main(args):

    # SAVE_DIR = f"/lustre/orion/csc590/scratch/jonathanlimsc/bgpt/data/math-adder" 
    reverse_ans = args.reverse
    ndigits = args.digits
    save_dir = args.save_dir + f"/{ndigits}-digit"


    if reverse_ans:
        print("Output number will be reversed.")
        save_dir += "-reversed"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ds = AdditionDatasetGenerator(ndigit=ndigits, test_ratio=0.2, reverse_answer=reverse_ans)
    ds.generate_problems(save_dir=save_dir)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an N-digit easy math problem e.g. 2-digit sum 23+12=35")
    parser.add_argument("--digits", type=int, required=True, default=2, help="Number of digits per number. Default is 2")
    parser.add_argument("--reverse", dest='reverse', action='store_true', help="Reverse answer number e.g. 23+12=53 instead of 35, due to left-to-right generation")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to the output directory to save train.txt and test.txt")
    parser.set_defaults(reverse=False)
    args = parser.parse_args()

    main(args)