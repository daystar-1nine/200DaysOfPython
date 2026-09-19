"""Generates a realistic, deterministic SMS Spam and Ham dataset for Day 101."""
import pandas as pd
import numpy as np

def generate_sms_dataset(filepath: str, n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    spam_templates = [
        "URGENT! Your mobile number was awarded a £2,000 bonus prize! Call 09061701461 immediately to claim.",
        "WINNER! As a valued customer you are selected to receive a £1000 cash reward. Text CLAIM to 87077.",
        "Congratulations! You won a FREE cruise to the Bahamas! Call now at 1-800-555-0199. T&Cs apply.",
        "PRIVATE! Your 2004 Account for 07781482378 has 800 points. Call 0871-872-9758 to claim your reward.",
        "Double your money in 24 hours! Guaranteed crypto returns. Click http://bit.ly/fast-crypto now!",
        "FREE RINGTONE! Reply YES to 80082 to get polyphonic ringtones directly to your phone. 1.50gbp/week.",
        "Claim your complimentary 2-week gym pass at FitnessFirst today! Text FIT to 88088.",
        "Dear customer, your bank account has been suspended due to suspicious activity. Verify at http://secure-bank-login.xyz",
        "Exclusive offer! 50% discount on luxury watches. Visit our website http://swiss-replica.co today only!",
        "URGENT: Call 09066362206 from landline. 150p/min. You have an unclaimed parcel waiting delivery.",
        "You have won a guaranteed £500 Amazon gift card! Claim now by replying GIFT.",
        "Loan approved! Up to £5000 deposited within 1 hour. No credit checks required! Call 08000930705.",
        "Hot singles in your area want to chat with you right now! Reply CHAT to 69696 to connect.",
        "Your mobile contract is up for renewal! Get a brand new iPhone 15 Pro for £0 upfront. Call 0800-111-222.",
        "WIN a £500 shopping spree at Tesco! To enter simply answer our quick 2-question survey: http://tesco-win.me",
        "Final notice! You have unpaid toll fees of £4.50. Pay immediately at http://highway-toll-uk.com to avoid legal action.",
        "Free entry in 2 a weekly competition to win £1000 cash. Text WIN to 82122 to enter now!",
        "Special promotion! Buy 1 get 2 free on all health supplements. Order at http://vitamins-direct.com.",
        "Alert: Unusual sign-in attempt detected on your PayPal account. If this was not you, visit http://paypal-verify.com.",
        "Your parcel delivery has failed because no one was home. Reschedule fee required: http://dpd-redelivery.org"
    ]

    ham_templates = [
        "Hey, are we still meeting for lunch at 1pm tomorrow?",
        "Can you please pick up some milk and eggs on your way home from work?",
        "Thanks for the lecture notes! They were super helpful for the exam prep.",
        "I'll be there in about 15 minutes, traffic is a bit slow today.",
        "Did you finish the Python assignment? Let's compare our answers tonight.",
        "Happy birthday! Wishing you a fantastic year ahead filled with joy and success!",
        "Sorry I missed your call earlier, I was in an important team meeting. What's up?",
        "Let's catch up this weekend. Are you free on Saturday afternoon?",
        "Don't forget we have the project standup meeting at 10am tomorrow morning.",
        "Great job on the presentation today, the client was really impressed with our slides!",
        "Can you send me the link to that recipe you made last week? It was delicious.",
        "Just reached the airport, flight is on time. See you soon!",
        "Where did you leave the spare car keys? I can't find them on the counter.",
        "Sounds good! Let's book the tickets for the 7pm movie screening.",
        "Could you review my pull request on GitHub when you have a moment?",
        "I'm feeling much better today, thanks for checking in on me!",
        "Let me know when you're free for a quick phone call to discuss the travel plans.",
        "The weather is supposed to be really nice this Sunday, want to go for a hike?",
        "I just sent over the revised budget spreadsheet. Please check if the numbers look right.",
        "Running a bit late, start eating without me. Be there around 7:30!"
    ]

    # Generate 85% ham, 15% spam (typical realistic class imbalance)
    n_spam = int(n_samples * 0.18)
    n_ham = n_samples - n_spam
    
    spam_data = np.random.choice(spam_templates, size=n_spam)
    ham_data = np.random.choice(ham_templates, size=n_ham)
    
    # Add slight realistic noise to templates
    noisy_spam = []
    for s in spam_data:
        if np.random.rand() > 0.5:
            s = s + f" Ref: #{np.random.randint(1000, 9999)}"
        noisy_spam.append(s)
        
    noisy_ham = []
    for h in ham_data:
        if np.random.rand() > 0.5:
            h = h + " :)"
        elif np.random.rand() > 0.7:
            h = h + " Thanks!"
        noisy_ham.append(h)
        
    df_spam = pd.DataFrame({"label": ["spam"] * n_spam, "text": noisy_spam})
    df_ham = pd.DataFrame({"label": ["ham"] * n_ham, "text": noisy_ham})
    
    df = pd.concat([df_spam, df_ham], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    df.to_csv(filepath, index=False, encoding="utf-8")
    return df

if __name__ == "__main__":
    from Day_101.app.config import config
