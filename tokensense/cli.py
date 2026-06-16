import sqlite3
import sys

import urllib.request
import os

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "report":
            print("TokenSense Spend Report")
            print("-----------------------")
            try:
                # We assume it's running in the directory where tokensense.db lives
                conn = sqlite3.connect("tokensense.db")
                cursor = conn.cursor()
                
                # Group by model
                cursor.execute("""
                    SELECT model, SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
                    FROM calls
                    GROUP BY model
                    ORDER BY SUM(cost_usd) DESC
                """)
                rows = cursor.fetchall()
                
                if not rows:
                    print("No API calls recorded yet.")
                    return
                    
                print(f"{'Model':<25} | {'Input Tokens':<15} | {'Output Tokens':<15} | {'Cost (USD)':<10}")
                print("-" * 75)
                
                total_cost = 0.0
                for row in rows:
                    model, in_tok, out_tok, cost = row
                    total_cost += cost
                    print(f"{model:<25} | {in_tok:<15} | {out_tok:<15} | ${cost:<10.4f}")
                    
                print("-" * 75)
                print(f"Total Spend: ${total_cost:.4f}")
                
            except sqlite3.OperationalError:
                print("No tokensense.db found in current directory. Have you made any API calls yet?")
        
        elif command == "update-prices":
            url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
            print(f"Downloading latest pricing from {url}...")
            
            try:
                import tokensense
                target_path = os.path.join(os.path.dirname(tokensense.__file__), "model_prices.json")
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    content = response.read()
                    
                with open(target_path, "wb") as f:
                    f.write(content)
                    
                print(f"Successfully updated model prices at: {target_path}")
            except Exception as e:
                print(f"Error updating prices: {e}")
                
        else:
            print(f"Unknown command: {command}")
            print("Usage: tokensense [report | update-prices]")
    else:
        print("Usage: tokensense [report | update-prices]")

if __name__ == "__main__":
    main()
