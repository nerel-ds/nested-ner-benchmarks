import torch
from pytorch_lightning import Trainer

from trainer import BertLabeling
from utils.get_parser import get_parser
# from utils.random_seed import set_random_seed

# set_random_seed(0)

def test():

    parser = get_parser() # Создание парсера командной строки в общем виде (см. get_parser)

    # Добавление аргументов командной строки, отвечающих самой модели
    parser = BertLabeling.add_model_specific_args(parser) 

    # Добавление всех возможных флагов Trainer (--gpus, --num_nodes и т.д.) из командной строки
    parser = Trainer.add_argparse_args(parser) 

    # Помощь по всем флагам командой строки - либо через -h / --help, либо (если указано pl.Trainer)
    # см. документацию по Trainer от Pytorch Lightning

    # Сохраняем все аргументы из командной строки
    args = parser.parse_args()

    model = BertLabeling(args) # Инициализиуем модель на их основе

    # Если грузим из чекпойнта
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint, 
                                         map_location=torch.device('cpu'))["state_dict"]) 

    trainer = Trainer.from_argparse_args(
        args,
        logger = False
    )
    data_loader = model.get_dataloader("dev", filter_tags=None)
    trainer.test(model, data_loader, ckpt_path = args.pretrained_checkpoint)

    tag_classes = ['AGE', 'AWARD', 'CITY', 'COUNTRY', 'CRIME', 'DATE', 'DISEASE', 'DISTRICT', 'EVENT', 'FACILITY', 
        'FAMILY', 'IDEOLOGY', 'LANGUAGE', 'LAW', 'LOCATION', 'MONEY', 'NATIONALITY', 'NUMBER', 'ORDINAL', 
        'ORGANIZATION', 'PERCENT', 'PERSON', 'PENALTY', 'PRODUCT', 'PROFESSION', 'RELIGION', 'STATE_OR_PROVINCE', # 'OUT', 
        'TIME', 'WORK_OF_ART'] 

    for tag in tag_classes:
        print(tag + ":")
        trainer.test(
            model, 
            model.get_dataloader("dev", filter_tags=[tag]), 
            ckpt_path=args.pretrained_checkpoint
        )
    #    print(tag + ":" + str(model.get_dataloader("test", tag = tag).__len__()))

if __name__ == '__main__':
    test() # None - тест по всем сущностям; иначе, выбрать сущность (например, tag_test='AGE')