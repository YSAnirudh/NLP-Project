import sys

from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer

# x = KnowledgeGraph(dataset='hello', custom_dataset_path='hello')
# x.prepare_data()
# print(x.dataset)
# trip = x.read_entities()
# # print(trip.h)
# for i in range(len(trip)):
#     print(trip[i])
# print(len(trip))

def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])
    print(args)
    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    model = model_def(**config.__dict__)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()


    trainer.infer_tails(1, 10, topk=5)
    trainer.infer_heads(10, 20, topk=5)
    trainer.infer_rels(18, 21, topk=5)

if __name__ == "__main__":
    main()
