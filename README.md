# TriMakeMore

A trigram-based name generator inspired by Andrej Karpathy's "makemore."

## Files

- **NNTrigram.py**: Trains a simple neural network using trigrams to generate names.
- **ContTrigram.py**: A statistical trigram model that generates names based on conditional probabilities.
- **names.txt**: A list of names used by both models.

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/nasseralbess/TriMakeMore.git
   cd TriMakeMore
   ```
2. Run the neural network model:
   ```bash
   python NNTrigram.py
   ``` 
3. Run the statistical model:
   ```bash
   python CountTrigram.py
   ```

## Output

Each script generates and prints new names based on the trained model.
