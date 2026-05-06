import pandas as pd
from pathlib import Path
from pykeen.triples import TriplesFactory

def transductive_split(path: str, ratios: list[float] = [0.8, 0.2], random_state: int = 42):
    df = pd.read_csv(path, usecols=["id_entity_1", "predicate", "id_entity_2"]).astype(str)
    triples = df[["id_entity_1", "predicate", "id_entity_2"]].to_numpy(dtype=str)
    tf = TriplesFactory.from_labeled_triples(
        triples=triples,
        create_inverse_triples=False,
        compact_id=True,
    )
    train, test = tf.split(ratios=ratios, random_state=random_state)
    return train, test

def save_factory_to_tsv(factory, path):
    mapped = factory.mapped_triples
    
    labeled = factory.label_triples(mapped)

    df = pd.DataFrame(
        labeled,
        columns=["id_entity_1", "predicate", "id_entity_2"]
    )

    df.to_csv(path, sep="\t", index=False, header=False)


def print_split_report(train_factory, test_factory):
    train_labeled = train_factory.label_triples(train_factory.mapped_triples)
    test_labeled = test_factory.label_triples(test_factory.mapped_triples)

    train_count = train_labeled.shape[0]
    test_count = test_labeled.shape[0]
    total_count = train_count + test_count

    all_nodes = train_factory.num_entities
    all_predicates = train_factory.num_relations

    print("\nSplit report:")
    print(f"Total triples: {total_count}")
    print(f"Train triples: {train_count}")
    print(f"Test triples: {test_count}")
    print(f"Total nodes: {all_nodes}")
    print(f"Total predicates: {all_predicates}\n")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
path = PROJECT_ROOT / "resources"
source_file = path / "triples.csv"
target_ratios = [0.8, 0.2]
seed = 42

train, test = transductive_split(source_file, ratios=target_ratios, random_state=seed)
save_factory_to_tsv(train, path / "train.tsv")
save_factory_to_tsv(test, path / "test.tsv")
print_split_report(train, test)