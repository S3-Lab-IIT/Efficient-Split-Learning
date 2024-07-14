from ImageSegmentation_Task.kits19.kits_trainer import PfslKits
from ImageSegmentation_Task.kits19.kits_nonkv_trainer import PfslKits as PfslKitsNonKV

from ImageSegmentation_Task.IXI.ixi_trainer import IXITrainer
from ImageSegmentation_Task.IXI.ixi_nonkv_trainer import IXITrainer as IXITrainerNonKV

from ImageSegmentation_Task.ISIC2019.isic_trainer import ISICTrainer

from ImageClassification_Task.ic_trainer import ICTrainer

from ImageSegmentation_Task.PCam.pcam_trainer import PCamTrainer

from ImageSegmentation_Task.COVID19.covid_trainer import CovidTrainer


from utils.argparser import parse_arguments
from pprint import pprint

if __name__ == '__main__':
    args = parse_arguments()

    pprint(vars(args))

    if args.dataset == 'kits19':
        if args.use_key_value_store:
            trainer = PfslKits(args)
        else:
            trainer = PfslKitsNonKV(args)

        trainer.fit()
        trainer.inference()

    elif args.dataset == 'IXI':
        if args.use_key_value_store:
            trainer = IXITrainer(args)
        else:
            trainer = IXITrainerNonKV(args)

        trainer.fit()
        trainer.inference()

    elif args.dataset == 'ISIC2019':
        if args.use_key_value_store:
            trainer = ISICTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        trainer.inference()

    elif args.dataset == 'PCam':
        if args.use_key_value_store:
            trainer = PCamTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'COVID19':
        if args.use_key_value_store:
            trainer = CovidTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        
    elif args.dataset == 'CIFAR10':
        if args.use_key_value_store:
            trainer = ICTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        # print("HYBRID INFERENCE")
        # trainer.inference_new()

    else:
        print('invalid dataset -_-')
