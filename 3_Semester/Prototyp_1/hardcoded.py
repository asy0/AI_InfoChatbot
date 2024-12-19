import spacy
from collections import Counter

nlp = spacy.load("de_core_news_lg") #  spaCy-Modell


def analyze_text_hardcoded():
    # Hardcodierter Text
    text = """
  Aschenputtel
        Es war einmal ein Mädchen, dem war die Mutter gestorben. Ihre neue Stiefmutter hatte zwei Töchter mit ins Haus gebracht und sie waren alle sehr gemein zu ihr. Sie musste den ganzen Tag schwer arbeiten, früh aufstehen, Wasser tragen, Feuer machen, kochen und waschen. Abends, wenn sie müde war, musste sie sich neben den Herd in die Asche legen. Und weil sie darum immer staubig und schmutzig war, nannten sie es Aschenputtel.
        Es begab sich, daß der König ein großes Fest veranstaltete, auf dem sich der Prinz eine Gemahlin aussuchen sollte. Aschenputtel bat die Stiefmutter, sie möchte ihm erlauben hinzugehen. „Aschenputtel,“ antwortete die Stiefmutter, „du bist voll Staub und Schmutz und willst zum Fest? Du darfst nicht mit, denn du hast keine prächtigen Kleider.“ Darauf eilte sie mit ihren stolzen Töchtern fort.
        Aschenputtel ging zum Grab ihrer Mutter und weinte bis die Tränen darauf herniederfielen. Als sie die Augen wieder öffnete, trug sie plötzlich ein prächtiges Kleid und goldene Schuhe.
        So ging es zum Fest und der Prinz tanzte mit ihr. Als es Abend war, wollte Aschenputtel fort, der Prinz wollte sie begleiten, aber sie entsprang ihm so geschwind, daß er nicht folgen konnte. Auf der Treppe verlor sie einen ihrer Schuhe. Der Prinz fand den Schuh und sprach: „Keine andere soll meine Gemahlin werden, als die, an deren Fuß dieser goldene Schuh paßt.“ Und er ließ im ganzen Königreich nach dem Mädchen suchen, der der Schuh passte.
        Als er zu ihrem Hause kam, da passte der Schuh wie angegossen. Und als Aschenputtel sich in die Höhe richtete und dem Prinzen ins Gesicht sah, da erkannte er sie.
        Und sie lebten glücklich alle Tage.
    """


    cleaned_text = " ".join(text.split())
   # print("Bereinigter Text:", cleaned_text)  # Debugging-Ausgabe
    doc = nlp(cleaned_text)

    # Gruppiere widersprüchliche Labels nach Text
    entity_counter = Counter([(ent.text, ent.label_) for ent in doc.ents])
    print("Entitäten gefunden:", entity_counter)  # Debugging-Ausgabe

    grouped_entities = {
        text: Counter(label for txt, label in entity_counter if txt == text).most_common(1)[0][0]
        for text, _ in entity_counter
    }

    # Extrahiere eindeutige Sätze
    unique_sentences = sorted(set(sent.text.strip() for sent in doc.sents), key=lambda x: x.lower())

    return {
        "entities": grouped_entities,
        "sentences": unique_sentences
    }


if __name__ == "__main__":
    # Ergebnisse der Analyse
    results = analyze_text_hardcoded()

    # Ausgabe der Entitäten
    print("\nEntitäten:")
    for entity, label in results["entities"].items():
        print(f" - {entity} ({label})")

    # Ausgabe der Sätze
    print("\nSätze:")
    for sentence in results["sentences"]:
        print(f" - {sentence}")
