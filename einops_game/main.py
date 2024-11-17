import einops
import random
import sys
import time
import torch

# So we don't need to type all the digits 
torch.set_printoptions(precision=1)    


ANSI_COLORS = [
    '\033[91m',  # Bright Red
    '\033[93m',  # Bright Yellow
    '\033[92m',  # Bright Green
    '\033[96m',  # Bright Cyan
    '\033[94m',  # Bright Blue
    '\033[95m',  # Bright Magenta
]
RESET = '\033[0m'
BOLD = '\033[1m'
BANNER = """
#######  ###  ##    ##  #######  ######  #######  ##      #######  ##   ##  ######
##       ###  ###   ##  ##   ##  ##  ##  ##   ##  ##      ##   ##  ##   ##  ##
#######  ###  ## #  ##  ##   ##  ######  ##   ##  ##      ##   ##  ##   ##  ######
##       ###  ##  # ##  ##   ##  ##      ##   ##  ##      ##   ##  ##   ##      ##
#######  ###  ##   ###  #######  ##      #######  ######  #######  #######  ######
"""

def print_rainbow_banner(speed=0.001):
    """Print the banner with rainbow colors using ANSI escape codes."""
    
    lines = BANNER.split('\n')
    for line in lines:
        if line.strip():  # Only process non-empty lines
            for i, char in enumerate(line):
                color = ANSI_COLORS[i % len(ANSI_COLORS)]
                sys.stdout.write(f"{color}{char}{RESET}")
                sys.stdout.flush()
                time.sleep(speed)
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\n')
    sys.stdout.write(RESET)  # Reset all colors at the end

class EinopsChallenge:
    def __init__(self):
        self.challenge_templates = [
            {
                'generator': lambda: torch.rand((2, 2)).round(decimals=1),  
                'operation': "rearrange(tensor, 'h w -> w h')",
                'hint': "Think about swapping height and width dimensions"
            },
            {
                'generator': lambda: torch.rand((1, 2, 2)).round(decimals=1),  
                'operation': "rearrange(tensor, 'b h w -> (b h) w')",
                'hint': "Combine the first two dimensions into one"
            },
            {
                'generator': lambda: torch.rand((1, 2)).round(decimals=1),  
                'operation': "repeat(tensor, 'h w -> h w c', c=2)",
                'hint': "Add a new dimension with 2 copies"
            },
            {
                'generator': lambda: torch.rand((2, 2, 2)).round(decimals=1),  
                'operation': "reduce(tensor, 'b h w -> b h 1', 'mean')",
                'hint': "Average the values in the width dimension (w)"
            },
            {
                'generator': lambda: torch.rand((2, 2, 2)).round(decimals=1),  
                'operation': "reduce(tensor, 'b h w -> b h', 'mean')",
                'hint': "Average the values in the width dimension (w)"
            },
            {
                'generator': lambda: torch.rand((2, 2)).round(decimals=1),  
                'operation': "repeat(tensor, 'h w -> h w c', c=2)",
                'hint': "Add a new dimension that repeats each element twice"
            }
        ]

    def format_tensor(self, tensor):
        """Format tensor for display, rounding to 1 decimal place if float"""
        if tensor.dtype in [torch.float32, torch.float64]:
            return str(tensor.round(decimals=1))
        return str(tensor)

    def get_challenge(self):
        """Return a random challenge with a freshly generated tensor"""
        template = random.choice(self.challenge_templates)
        return {
            'tensor': template['generator'](),  # Generate a new random tensor
            'operation': template['operation'],
            'hint': template['hint']
        }
    
    def check_answer(self, challenge, user_answer):
        """Check if user's answer matches the expected output"""
        try:
            # Get the actual result using einops
            tensor = challenge['tensor']
            expected = eval(f"einops.{challenge['operation']}")
            
            # Convert user's answer to tensor
            # First, evaluate the string as a Python expression
            try:
                user_tensor = eval(user_answer)
                if not isinstance(user_tensor, torch.Tensor):
                    return False, f"Your answer should be a PyTorch tensor. The answer is\n{expected}"
            except:
                return False, f"Invalid tensor format. The answer is\n{expected}"

            # Check if shapes match
            if user_tensor.shape != expected.shape:
                return False, f"Shape mismatch! Expected shape {expected.shape}, got {user_tensor.shape}. The answer is {expected}"

            # Check if values match
            if not torch.allclose(user_tensor, expected, rtol=1e-3):
                return False, f"Values don't match the expected result. The answer is {expected}"

            return True, "Correct!"
            
        except Exception as e:
            return False, f"Error evaluating answer: {str(e)}"
    
    def game_over(self, score, total):
        print("\n" + "="*50)
        print("\n🎮 GAME OVER! 🎮")
        print(f"\nFinal Score: {score}/{total} ({(score/total)*100:.1f}%)")
        # Different messages based on performance
        if score == total:
            print("\n🏆 Perfect score! You're an Einops Master! 🏆")
        elif score >= total * 0.8:
            print("\n🌟 Excellent! You're well on your way to Einops mastery!")
        elif score >= total * 0.6:
            print("\n👍 Good job! Keep practicing to improve your Einops skills!")
        else:
            print("\n📚 lol, damn.")


def play_ball():
    print("\n🎮 Welcome to Einopolous!")
    print("Try to predict the output of einops operations on PyTorch tensors.")
    print("Enter your answer as a PyTorch tensor (e.g., torch.tensor([[0.1, 0.2], [0.3, 0.4]]))")    
    print(f"{BOLD}FORMATTING MATTERS{RESET}. Make sure your tensors have correct spacing.")
    print("Type 'hint' for a hint, or 'q' to exit.\n")
    

    game = EinopsChallenge()
    score = 0
    total = 0    
    max_questions = 10
    while True:
        challenge = game.get_challenge()
        total += 1
        
        print("\n" + "="*50)
        print(f"{BOLD}Challenge {total}:{RESET}")        
        print(f"\nInput tensor:")
        print(game.format_tensor(challenge['tensor']))
        print(f"\nEinops operation:")
        print(f"einops.{challenge['operation']}")
        
        while True:
            answer = input("\nYour answer (or 'hint', 'q'): ").strip()
            
            if answer.lower() == 'q':
                game.game_over(score, total)                
                return
            
            if answer.lower() == 'hint':
                print(f"\nHint: {challenge['hint']}")
                continue
                
            is_correct, message = game.check_answer(challenge, answer)            
            
            if is_correct:
                print(f"\n{message}")                                
                score += 1                
                break
            else:
                print(f"\n{message}")                
                break

        print(f"\nCurrent score: {score}/{total}")

        if total == max_questions:
            game.game_over(score, total)        
            break
            
def main():    
    print_rainbow_banner()
    play_ball()

if __name__ == "__main__":
    main()
