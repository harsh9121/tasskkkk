import json
import random
import string
from pathlib import Path

FIRST_NAMES = ["john", "maria", "arjun", "sara", "li", "fatima"]
LAST_NAMES = ["doe", "patel", "khan", "singh", "garcia", "kim"]
CITIES = ["new york", "san francisco", "mumbai", "london", "berlin"]
LOCATIONS = ["central park", "times square", "india gate", "tower bridge"]

DIGITS_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

def random_credit_card():
    return "".join(random.choice(string.digits) for _ in range(16))

def spell_out_number(num_str):
    return " ".join(DIGITS_WORDS[d] for d in num_str)

def random_phone_digits():
    length = random.choice([10, 11, 12])
    return "".join(random.choice(string.digits) for _ in range(length))

def random_email(name=None):
    if name:
        user = name.replace(" ", "")
    else:
        user = "".join(random.choices(string.ascii_lowercase, k=7))
    domain = random.choice(["gmail", "yahoo", "outlook", "company"])
    tld = random.choice(["com", "org", "net", "io"])
    return f"{user}@{domain}.{tld}"

def email_to_noisy(email):
    user, domain = email.split("@")
    dom, tld = domain.split(".")
    return f"{user} at {dom} dot {tld}"

def random_date_text():
    days = ["first", "second", "third", "fifteenth", "twenty first",
            "twenty fifth", "thirty first"]
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    years = ["twenty nineteen", "twenty twenty", "twenty twenty one",
             "twenty twenty three", "twenty twenty four"]
    return f"{random.choice(days)} of {random.choice(months)} {random.choice(years)}"

TEMPLATES = [
    "my credit card number is {cc_words} please do not share it",
    "the card is {cc_words} uh and the name is {name}",
    "call me on {phone_words} tomorrow",
    "you can email me at {email_noisy}",
    "i will come on {date_text}",
    "i live in {city} near {location}",
    "ship this to {name} in {city}",
    "{name} email {email_noisy} phone {phone_words}",
]

def generate_example(example_id: int):
    name = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
    city = random.choice(CITIES)
    location = random.choice(LOCATIONS)
    cc = random_credit_card()
    cc_words = spell_out_number(cc)
    phone = random_phone_digits()
    phone_words = spell_out_number(phone)
    email = random_email(name=name)
    email_noisy = email_to_noisy(email)
    date_text = random_date_text()

    tpl = random.choice(TEMPLATES)

    text = tpl.format(
        cc_words=cc_words,
        name=name,
        phone_words=phone_words,
        email_noisy=email_noisy,
        date_text=date_text,
        city=city,
        location=location,
    )

    entities = []

    def add_span(span_text, label):
        start = text.index(span_text)
        end = start + len(span_text)
        entities.append({"start": start, "end": end, "label": label})

    if "{cc_words}" in tpl:
        add_span(cc_words, "CREDIT_CARD")
    if "{phone_words}" in tpl:
        add_span(phone_words, "PHONE")
    if "{email_noisy}" in tpl:
        add_span(email_noisy, "EMAIL")
    if "{date_text}" in tpl:
        add_span(date_text, "DATE")
    if "{name}" in tpl:
        add_span(name, "PERSON_NAME")
    if "{city}" in tpl:
        add_span(city, "CITY")
    if "{location}" in tpl:
        add_span(location, "LOCATION")

    return {
        "id": f"synt_{example_id:04d}",
        "text": text,
        "entities": entities,
    }

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    train_examples = [generate_example(i) for i in range(800)]
    dev_examples = [generate_example(800 + i) for i in range(150)]

    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(out_dir / "dev.jsonl", "w") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()
