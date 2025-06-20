import random
import string

KEYWORDS = ["python", "bot", "vex", "script", "client", "agent", "crawler", "utility", "tool", "data", "validation", "service", "monitor", "engine"]

def generate_random_email():
    name_length = random.randint(5, 12)
    domain_length = random.randint(5, 10)
    
    name_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=name_length))
    domain_part = ''.join(random.choices(string.ascii_lowercase, k=domain_length))
    
    tlds = ["com", "org", "net", "io", "dev", "ai", "tech", "app", "live", "online"]
    tld = random.choice(tlds)
    
    return f"{name_part}@{domain_part}.{tld}"


def generate_user_agent() -> str:
    email = generate_random_email()
    
    base = f""
    
    for word in KEYWORDS:
        if random.randint(0, 1) == 1:
            base += f"{word }"
    base += f"/{random.randint(0, 10)}.{random.randint(0, 25)} {email}"
    
    return base
        
        
def generate_header(api_key: str) -> dict:
    user_agent = generate_user_agent()
    
    
    HEADERS = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "accept-language": "en",
        "user-agent": user_agent
    }
    
    return HEADERS
    
    
    

