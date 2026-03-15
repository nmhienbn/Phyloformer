import math
import os
import random
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import PurePath
from pprint import pprint
from random import choice
from time import time
from typing import Any, Dict, Iterable, Optional, Type, Union

import numpy as np
import torch
import wandb
from torch.distributed import (
    ReduceOp,
    all_gather_into_tensor,
    all_reduce,
    barrier,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .data import (
    PADDING_TOKEN,
    CheckpointableDsitributedSampler,
    CheckpointableRandomSampler,
    CheckpointableSequentialSampler,
    PhyloDataset,
    make_pairs,
)
from .modules import PhyloformerSeq
from .utils import instantiate_enum, val_to_toml

# Name generation utilities
# fmt: off
ANIMALS="aardvark,abyssinian,affenpinscher,akbash,akita,albatross,alligator,angelfish,angora,ant,anteater,antelope,argentino,armadillo,audemer,avocet,axolotl,aye,baboon,badger,balinese,bandicoot,barb,barnacle,barracuda,bat,beagle,bear,beaver,bee,bee-eater,beetle,bernard,binturong,bird,birman,bison,bloodhound,boar,bobcat,bombay,bongo,bonobo,booby,bordeaux,bracke,budgerigar,buffalo,bulldog,bullfrog,burmese,butterfly,buzzard,caiman,camel,capuchin,capybara,caracal,cassowary,cat,caterpillar,catfish,cattle,centipede,chameleon,chamois,cheetah,chicken,chihuahua,chimpanzee,chin,chinchilla,chinook,chipmunk,chow,cichlid,civet,clam,coati,cockroach,collie,coon,coonhound,corgi,cougar,cow,coyote,crab,crane,crocodile,cuscus,cuttlefish,dachsbracke,dachshund,dalmatian,dane,deer,devil,dhole,dingo,discus,dodo,dog,dogfish,dollar,dolphin,donkey,dormouse,dragon,dragonfly,drever,duck,dugong,dunker,eagle,earwig,echidna,eel,elephant,eleuth,emperor,emu,falcon,ferret,fish,flamingo,flounder,fly,forest,fossa,fousek,fowl,fox,foxhound,frigatebird,frise,frog,gar,gecko,gerbil,gharial,gibbon,giraffe,goat,goose,gopher,gorilla,grasshopper,greyhound,grouse,guppy,hamster,hare,harrier,havanese,hedgehog,heron,himalayan,hippopotamus,hornet,horse,hound,human,hummingbird,husky,hyena,hyrax,ibis,iguana,impala,indri,insect,jackal,jaguar,javanese,jellyfish,kakapo,kangaroo,kingfisher,kiwi,koala,kudu,labradoodle,ladybird,lemming,lemur,leopard,liger,lion,lionfish,lizard,llama,lobster,loon,lynx,macaque,macaw,magpie,malamute,maltese,mammoth,manatee,mandrill,markhor,marmoset,mastiff,mau,mayfly,meerkat,millipede,mist,mole,molly,mongoose,mongrel,monkey,monster,moorhen,moose,moth,mouse,mule,neanderthal,newfoundland,newt,nightingale,numbat,ocelot,octopus,okapi,olm,opossum,orang-utan,oriole,ostrich,otter,owl,oyster,pademelon,panda,panther,paradise,parrot,peacock,peccary,pekingese,pelican,penguin,persian,pheasant,pig,pika,pike,pinscher,piranha,platypus,pointer,poodle,porcupine,possum,prawn,puffin,pug,puma,quail,quetzal,quokka,quoll,rabbit,raccoon,ragdoll,rat,rattlesnake,ray,reindeer,retriever,rhinoceros,robin,rottweiler,russel,salamander,saola,schnauzer,scorpion,seahorse,seal,serval,setter,shark,sheep,sheepdog,shepherd,shrew,shrimp,siamese,siberian,skater,skunk,sloth,slug,snail,snake,snowshoe,somali,spaniel,sparrow,spider,spitz,sponge,spoonbill,squid,squirrel,squirt,starfish,stingray,stoat,swan,tamarin,tang,tapir,tarantula,tarsier,termite,terrier,tetra,tiffany,tiger,toad,tortoise,toucan,tropicbird,tuatara,turkey,turtle,tzu,uakari,uguisu,umbrellabird,urchin,vole,vulture,wallaby,walrus,warthog,wasp,weasel,whale,whippet,wildebeest,wolf,wolfhound,wolverine,wombat,woodlouse,woodpecker,worm,wrasse,yak,zebra,zebu,zonkey,zorse".split(",")
ADJECIVES="abandoned,able,absolute,academic,acceptable,acclaimed,accomplished,accurate,aching,acidic,acrobatic,active,actual,adept,admirable,admired,adolescent,adorable,adorable,adored,advanced,adventurous,affectionate,afraid,aged,aggravating,aggressive,agile,agitated,agonizing,agreeable,ajar,alarmed,alarming,alert,alienated,alive,all,altruistic,amazing,ambitious,ample,amused,amusing,anchored,ancient,angelic,angry,anguished,animated,annual,another,antique,anxious,any,apprehensive,appropriate,apt,arctic,arid,aromatic,artistic,ashamed,assured,astonishing,athletic,attached,attentive,attractive,austere,authentic,authorized,automatic,avaricious,average,aware,awesome,awful,awkward,babyish,back,bad,baggy,bare,barren,basic,beautiful,belated,beloved,beneficial,best,better,bewitched,big,biodegradable,bitter,bland,blank,blaring,bleak,blind,blissful,blond,blushing,bogus,boiling,bold,bony,boring,bossy,both,bouncy,bountiful,bowed,brave,breakable,brief,bright,brilliant,brisk,broken,bruised,bubbly,bulky,bumpy,buoyant,burdensome,burly,bustling,busy,buttery,buzzing,calculating,calm,candid,canine,capital,carefree,careful,careless,caring,cautious,cavernous,celebrated,charming,cheap,cheerful,cheery,chief,chilly,chubby,circular,classic,clean,clear,clever,close,closed,cloudy,clueless,clumsy,cluttered,coarse,cold,colorful,colorless,colossal,comfortable,common,compassionate,competent,complete,complex,complicated,composed,concerned,concrete,confused,conscious,considerate,constant,content,conventional,cooked,cool,cooperative,coordinated,corny,corrupt,costly,courageous,courteous,crafty,crazy,creamy,creative,creepy,criminal,crisp,critical,crooked,crowded,cruel,crushing,cuddly,cultivated,cultured,cumbersome,curly,curvy,cute,cylindrical,damaged,damp,dangerous,dapper,daring,dark,darling,dazzling,dead,deadly,deafening,dear,dearest,decent,decimal,decisive,deep,defenseless,defensive,defiant,deficient,definite,definitive,delayed,delectable,delicious,delightful,delirious,demanding,dense,dental,dependable,dependent,descriptive,deserted,detailed,determined,devoted,different,difficult,digital,diligent,dim,dimpled,dimwitted,direct,dirty,disastrous,discrete,disfigured,disguised,disgusting,dishonest,disloyal,dismal,dismal,distant,distant,distinct,distorted,dizzy,dopey,doting,double,downright,downright,drab,drafty,dramatic,dreary,dreary,droopy,dry,dual,dull,dutiful,each,eager,early,earnest,easy,ecstatic,edible,educated,elaborate,elastic,elated,elderly,electric,elegant,elementary,elliptical,embarrassed,embellished,eminent,emotional,empty,enchanted,enchanting,energetic,enlightened,enormous,enraged,entire,envious,equal,equatorial,essential,esteemed,ethical,euphoric,even,evergreen,everlasting,every,evil,exalted,excellent,excitable,excited,exciting,exemplary,exhausted,exotic,expensive,experienced,expert,extraneous,extroverted,fabulous,failing,faint,fair,faithful,fake,false,familiar,famous,fancy,fantastic,far,faraway,fast,fat,fatal,fatherly,favorable,favorite,fearful,fearless,feisty,feline,female,feminine,few,fickle,filthy,fine,finished,firm,first,firsthand,fitting,fixed,flaky,flamboyant,flashy,flat,flawed,flawless,flickering,flimsy,flippant,flowery,fluffy,fluid,flustered,focused,fond,foolhardy,foolish,forceful,forked,formal,forsaken,forthright,fortunate,fragrant,frail,frank,frayed,free,french,frequent,fresh,friendly,frightened,frightening,frigid,frilly,frivolous,frizzy,front,frosty,frozen,frugal,fruitful,full,fumbling,functional,funny,fussy,fuzzy,gargantuan,gaseous,general,generous,gentle,genuine,giant,giddy,gifted,gigantic,giving,glamorous,glaring,glass,gleaming,gleeful,glistening,glittering,gloomy,glorious,glossy,glum,golden,good,gorgeous,graceful,gracious,grand,grandiose,granular,grateful,grave,great,greedy,gregarious,grim,grimy,gripping,grizzled,gross,grotesque,grouchy,grounded,growing,growling,grown,grubby,gruesome,grumpy,guilty,gullible,gummy,hairy,half,handmade,handsome,handy,happy,hard,harmful,harmless,harmonious,harsh,hasty,hateful,haunting,healthy,heartfelt,hearty,heavenly,heavy,hefty,helpful,helpless,hidden,hideous,high,hilarious,hoarse,hollow,homely,honest,honorable,honored,hopeful,horrible,hospitable,hot,huge,humble,humiliating,humming,humongous,hungry,hurtful,husky,icky,icy,ideal,idealistic,identical,idiotic,idle,idolized,ignorant,ill,illegal,illiterate,illustrious,imaginary,imaginative,immaculate,immaterial,immediate,immense,impartial,impassioned,impeccable,imperfect,imperturbable,impish,impolite,important,impossible,impractical,impressionable,impressive,improbable,impure,inborn,incomparable,incompatible,incomplete,inconsequential,incredible,indelible,indolent,inexperienced,infamous,infantile,infatuated,inferior,infinite,informal,innocent,insecure,insidious,insignificant,insistent,instructive,insubstantial,intelligent,intent,intentional,interesting,internal,international,intrepid,ironclad,irresponsible,irritating,itchy,jaded,jagged,jaunty,jealous,jittery,joint,jolly,jovial,joyful,joyous,jubilant,judicious,juicy,jumbo,jumpy,junior,juvenile,kaleidoscopic,keen,key,kind,kindhearted,kindly,klutzy,knobby,knotty,knowing,knowledgeable,known,kooky,kosher,lame,lanky,large,last,lasting,late,lavish,lawful,lazy,leading,leafy,lean,left,legal,legitimate,light,lighthearted,likable,likely,limited,limp,limping,linear,lined,liquid,little,live,lively,livid,loathsome,lone,lonely,long,loose,lopsided,lost,loud,lovable,lovely,loving,low,loyal,lucky,lumbering,luminous,lumpy,lustrous,luxurious,mad,magnificent,majestic,major,male,mammoth,married,marvelous,masculine,massive,mature,meager,mealy,mean,measly,meaty,medical,mediocre,medium,meek,mellow,melodic,memorable,menacing,merry,messy,metallic,mild,milky,mindless,miniature,minor,minty,miserable,miserly,misguided,misty,mixed,modern,modest,moist,monstrous,monthly,monumental,moral,mortified,motherly,motionless,mountainous,muddy,muffled,multicolored,mundane,murky,mushy,musty,muted,mysterious,naive,narrow,nasty,natural,naughty,nautical,near,neat,necessary,needy,negative,neglected,negligible,neighboring,nervous,new,next,nice,nifty,nimble,nippy,nocturnal,noisy,nonstop,normal,notable,noted,noteworthy,novel,noxious,numb,nutritious,nutty,obedient,obese,oblong,oblong,obvious,occasional,odd,oddball,offbeat,offensive,official,oily,old,only,open,optimal,optimistic,opulent,orderly,ordinary,organic,original,ornate,ornery,other,our,outgoing,outlandish,outlying,outrageous,outstanding,oval,overcooked,overdue,overjoyed,overlooked,palatable,pale,paltry,parallel,parched,partial,passionate,past,pastel,peaceful,peppery,perfect,perfumed,periodic,perky,personal,pertinent,pesky,pessimistic,petty,phony,physical,piercing,pitiful,plain,plaintive,plastic,playful,pleasant,pleased,pleasing,plump,plush,pointed,pointless,poised,polished,polite,political,poor,popular,portly,posh,positive,possible,potable,powerful,powerless,practical,precious,precious,present,prestigious,pretty,previous,pricey,prickly,primary,prime,pristine,private,prize,probable,productive,profitable,profuse,proper,proud,prudent,punctual,pungent,puny,pure,pushy,putrid,puzzled,puzzling,quaint,qualified,quarrelsome,quarterly,queasy,querulous,questionable,quick,quiet,quintessential,quirky,quixotic,quizzical,radiant,ragged,rapid,rare,rash,raw,ready,real,realistic,reasonable,recent,reckless,rectangular,reflecting,regal,regular,reliable,relieved,remarkable,remorseful,remote,repentant,repulsive,required,respectful,responsible,revolving,rewarding,rich,right,rigid,ringed,ripe,roasted,robust,rosy,rotating,rotten,rough,round,rowdy,royal,rubbery,ruddy,rude,rundown,runny,rural,rusty,sad,safe,salty,same,sandy,sane,sarcastic,sardonic,satisfied,scaly,scarce,scared,scary,scented,scholarly,scientific,scornful,scratchy,scrawny,second,secondary,secret,selfish,sentimental,separate,serene,serious,serpentine,several,severe,shabby,shadowy,shady,shallow,shameful,shameless,sharp,shimmering,shiny,shocked,shocking,shoddy,short,showy,shrill,shy,sick,silent,silky,silly,similar,simple,simplistic,sinful,single,sizzling,skeletal,skinny,sleepy,slight,slim,slimy,slippery,slow,slushy,small,smart,smoggy,smooth,smug,snappy,snarling,sneaky,sniveling,snoopy,sociable,soft,soggy,solid,somber,some,sophisticated,sore,sorrowful,soulful,soupy,sour,spanish,sparkling,sparse,specific,spectacular,speedy,spherical,spicy,spiffy,spirited,spiteful,splendid,spotless,spotted,spry,square,squeaky,squiggly,stable,staid,stained,stale,standard,starchy,stark,starry,steel,steep,sticky,stiff,stimulating,stingy,stormy,straight,strange,strict,strident,striking,striped,strong,studious,stunning,stupendous,stupid,sturdy,stylish,subdued,submissive,substantial,subtle,suburban,sudden,sugary,sunny,super,superb,superficial,superior,supportive,surprised,suspicious,svelte,sweaty,sweet,sweltering,swift,sympathetic,talkative,tall,tame,tangible,tart,tasty,tattered,taut,tedious,teeming,tempting,tender,tense,tepid,terrible,terrific,testy,thankful,that,these,thick,thin,third,thirsty,this,thorny,thorough,those,thoughtful,threadbare,thrifty,thunderous,tidy,tight,timely,tinted,tiny,tired,torn,total,tough,tragic,trained,traumatic,treasured,tremendous,tremendous,triangular,tricky,trifling,trim,trivial,troubled,true,trusting,trustworthy,trusty,truthful,tubby,turbulent,twin,ugly,ultimate,unacceptable,unaware,uncomfortable,uncommon,unconscious,understated,unequaled,uneven,unfinished,unfit,unfolded,unfortunate,unhappy,unhealthy,uniform,unimportant,unique,united,unkempt,unknown,unlawful,unlined,unlucky,unnatural,unpleasant,unrealistic,unripe,unruly,unselfish,unsightly,unsteady,unsung,untidy,untimely,untried,untrue,unused,unusual,unwelcome,unwieldy,unwilling,unwitting,unwritten,upbeat,upright,upset,urban,usable,used,useful,useless,utilized,utter,vacant,vague,vain,valid,valuable,vapid,variable,vast,velvety,venerated,vengeful,verifiable,vibrant,vicious,victorious,vigilant,vigorous,villainous,violent,virtual,virtuous,visible,vital,vivacious,vivid,voluminous,wan,warlike,warm,warmhearted,warped,wary,wasteful,watchful,waterlogged,watery,wavy,weak,wealthy,weary,webbed,wee,weekly,weepy,weighty,weird,welcome,wet,which,whimsical,whirlwind,whispered,whole,whopping,wicked,wide,wiggly,wild,willing,wilted,winding,windy,winged,wiry,wise,witty,wobbly,woeful,wonderful,wooden,woozy,wordy,worldly,worn,worried,worrisome,worse,worst,worthless,worthwhile,worthy,wrathful,wretched,writhing,wrong,wry,yawning,yearly,yellowish,young,youthful,yummy,zany,zealous,zesty,zigzag".split(",")
COLORS="alizarin,amaranth,amber,amethyst,apricot,aqua,aquamarine,asparagus,auburn,azure,beige,bistre,black,blue,brass,bronze,brown,buff,burgundy,cardinal,carmine,celadon,cerise,cerulean,champagne,charcoal,chartreuse,chestnut,chocolate,cinnabar,cinnamon,cobalt,copper,coral,corn,cornflower,cream,crimson,cyan,dandelion,denim,ecru,emerald,eggplant,firebrick,flax,fuchsia,gamboge,gold,goldenrod,green,gray,harlequin,heliotrope,indigo,ivory,jade,khaki,lavender,lemon,lilac,lime,linen,magenta,magnolia,malachite,maroon,mauve,mustard,myrtle,ochre,olive,olivine,orange,orchid,peach,pear,periwinkle,persimmon,pink,platinum,plum,puce,pumpkin,purple,razzmatazz,red,rose,ruby,russet,rust,saffron,salmon,sangria,sapphire,scarlet,seashell,sepia,silver,smalt,tan,tangerine,taupe,teal,thistle,tomato,turquoise,ultramarine,vermilion,violet,viridian,wheat,white,wisteria,yellow,zucchini".split(",")
# Pretty header for logging
HEADER="""\
░▒█▀▀█░█░░░░█░░█░█░░▄▀▀▄░▒█▀▀▀░▄▀▀▄░█▀▀▄░█▀▄▀█░█▀▀░█▀▀▄
░▒█▄▄█░█▀▀█░█▄▄█░█░░█░░█░▒█▀▀░░█░░█░█▄▄▀░█░▀░█░█▀▀░█▄▄▀
░▒█░░░░▀░░▀░▄▄▄▀░▀▀░░▀▀░░▒█░░░░░▀▀░░▀░▀▀░▀░░▒▀░▀▀▀░▀░▀▀\
"""
# fmt: on


def generate_run_name():
    time = datetime.today().strftime("%Y%m%d-%H%M%S")
    return f"{time}-{choice(COLORS)}-{choice(ADJECIVES)}-{choice(ANIMALS)}"


class DistributedLogFile:
    def __init__(self, path: str, rank: int, buffering: int = -1):
        self.path = path
        self.rank = rank
        self.buf = buffering
        self.file_handle = None

    def __enter__(self):
        if self.rank == 0:
            self.file_handle = open(self.path, mode="w", buffering=self.buf)
        return self

    def __exit__(self, *_):
        if self.rank == 0 and self.file_handle is not None:
            self.file_handle.flush()
            self.file_handle.close()

    def write(self, content: str):
        if self.rank == 0 and self.file_handle is not None:
            self.file_handle.write(content)

    def writelines(self, lines: Iterable[str]):
        if self.rank == 0 and self.file_handle is not None:
            self.file_handle.writelines(lines)


class DistributedProgressBar:
    def __init__(
        self, total: int, desc: str, rank: int, leave: bool = True, initial: int = 0
    ):
        self.rank = rank
        self.total = total
        self.initial = initial
        self.desc = desc
        self.pbar = None
        self.leave = leave

    def __enter__(self):
        if self.rank == 0:
            self.pbar = tqdm(
                total=self.total, initial=self.initial, desc=self.desc, leave=self.leave
            )
        return self

    def __exit__(self, *_):
        if self.rank == 0 and self.pbar is not None:
            self.pbar.close()

    def update(self, value: int):
        if self.rank == 0 and self.pbar is not None:
            self.pbar.update(value)


class OptType(Enum):
    """Optimizers you can use"""

    ADAM = auto()
    ADAMW = auto()
    SGD = auto()
    # ... add more here and in init

    def init(self, params, lr, **kwargs) -> Optimizer:
        cls = OptType

        if self == cls.ADAM:
            return Adam(params, lr=lr, **kwargs)
        elif self == cls.ADAMW:
            return AdamW(params, lr=lr, **kwargs)
        elif self == cls.SGD:
            return SGD(params, lr=lr, **kwargs)

        raise NotImplementedError(
            "The optimizer you are trying to use is not implemented"
        )


def _mask_and_denom(y, y_hat, y_mask):
    # Apply mask if needed
    if y_mask is not None:
        y = y.masked_fill(~y_mask, 0)
        y_hat = y_hat.masked_fill(~y_mask, 0)

    denom = y_mask if y_mask is not None else torch.ones_like(y_hat)

    return y, y_hat, denom


def l1(y, y_hat, y_mask=None):
    y, y_hat, denom = _mask_and_denom(y, y_hat, y_mask)
    return ((y - y_hat).abs().sum(dim=-1) / denom.sum(dim=-1)).mean()


def l2(y, y_hat, y_mask=None):
    y, y_hat, denom = _mask_and_denom(y, y_hat, y_mask)
    return ((y - y_hat).pow(2).sum(dim=-1) / denom.sum(dim=-1)).mean()


def rmse(y, y_hat, y_mask=None):
    mse = l2(y, y_hat, y_mask)
    return mse.sqrt()


def mre(y, y_hat, y_mask=None):
    y, y_hat, denom = _mask_and_denom(y, y_hat, y_mask)
    y_denom = y
    # Fill padding spots with 1s to avoid division by 0 if padded
    if y_mask is not None:
        y_denom = y.masked_fill(~y_mask, 1)
    return (((y - y_hat) / y_denom).abs().sum(dim=-1) / denom.sum(dim=-1)).mean()


class LossType(Enum):
    """Loss functions you can use"""

    L1 = auto()
    MAE = auto()
    L2 = auto()
    MSE = auto()
    RMSE = auto()
    MRE = auto()
    # .. Add more here and in __call__

    def __call__(self, y, y_hat, y_mask=None):
        if self == LossType.L1 or self == LossType.MAE:
            return l1(y, y_hat, y_mask)
        elif self == LossType.L2 or self == LossType.MSE:
            return l2(y, y_hat, y_mask)
        elif self == LossType.MRE:
            return mre(y, y_hat, y_mask)
        elif self == LossType.RMSE:
            return rmse(y, y_hat, y_mask)

        raise NotImplementedError(
            "The loss value you are trying to call is not implemented"
        )


class SchedulerType(Enum):
    """Usable LR schedulers"""

    CONSTANT = auto()
    LINEAR_DECAY = auto()
    # ... Add more here and in init

    def init(self, optimizer, total_steps: int, warmup: int = 0) -> LambdaLR:
        cls = SchedulerType
        if self == cls.CONSTANT:
            if warmup == 0:
                return get_constant_schedule(optimizer)
            else:
                return get_constant_schedule_with_warmup(
                    optimizer, num_warmup_steps=warmup
                )
        elif self == cls.LINEAR_DECAY:
            if warmup == 0:
                return get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps,
                )
            else:
                return get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps,
                )

        raise NotImplementedError(
            "The scheduler you are trying to use is not implemented"
        )


class MaxStepLimiter:
    def __init__(self, max_steps: Optional[int] = None) -> None:
        self.max_steps = max_steps

    def should_stop(self, step) -> tuple[bool, str]:
        if self.max_steps is None:
            return False, ""

        return step >= self.max_steps, f"Reached Max number of steps={self.max_steps}"


class EarlyStopper:
    def __init__(self, patience: Optional[int] = None, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.no_improvement_counter = 0
        self.min_loss = None

    def should_stop(self, loss) -> tuple[bool, str]:
        if self.patience is None:  # Checker is disabled
            return (False, "")
        if self.min_loss is None:
            self.min_loss = loss
        elif loss + self.delta < self.min_loss:  # Improved by at least delta
            self.min_loss = loss
            self.no_improvement_counter = 0
        else:
            self.no_improvement_counter += 1
            if self.no_improvement_counter >= self.patience:
                return (True, f"No improvement after {self.patience} checks")

        return False, ""

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            patience=self.patience,
            delta=self.delta,
            counter=self.no_improvement_counter,
            min_loss=self.min_loss,
        )

    @staticmethod
    def from_state_dict(state: Dict[str, Any]) -> "EarlyStopper":
        stopper = EarlyStopper(state["patience"], state["delta"])
        stopper.no_improvement_counter = state["counter"]
        stopper.min_loss = state["min_loss"]

        return stopper


class LossChecker:
    def __init__(
        self,
        threshold: Optional[float] = None,
        check_inf_nan: bool = True,
        warmup: int = 5,
    ):
        self.threshold = threshold
        self.check_inf_nan = check_inf_nan
        self.warmup = warmup
        self.iter = 0

    def should_stop(self, loss) -> tuple[bool, str]:
        # Do not check during first steps
        if self.iter < self.warmup:
            self.iter += 1
            return False, ""

        if self.check_inf_nan and (torch.isnan(loss) or torch.isinf(loss)):
            return True, "Anomaly detected in loss (NaN or infinity)"

        if self.threshold:
            return (
                loss > self.threshold,
                f"Loss={loss} exceeded set threshold of {self.threshold}",
            )

        return False, ""

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            threshold=self.threshold,
            check_inf_nan=self.check_inf_nan,
            warmup=self.warmup,
            iter=self.iter,
        )

    @staticmethod
    def from_state_dict(state: Dict[str, Any]) -> "LossChecker":
        iter = state.pop("iter")
        checker = LossChecker(**state)
        checker.iter = iter
        return checker


def seed_everything(seed):
    """Seed all random generators"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        optimizer_type: OptType,
        loss_func: LossType,
        n_epochs: int,
        train_path: str,
        val_path: str,
        train_bs: int,
        val_bs: int,
        seed: int,
        device: Union[torch.device, str],
        rank: int,
        n_devices: int,
        val_every: int,
        log_every: int,
        scheduler_type: SchedulerType,
        warmup_steps: int,
        project_name: str,
        run_name: str,
        optimizer_kwargs: dict[str, Any] = {},
        artefacts: list[tuple[str, str]] = [],  # List of artefacts to log
        max_steps: Optional[int] = None,
        max_loss_threshold: Optional[float] = None,
        check_loss_anomalies: bool = True,
        loss_checker_warmup: int = 0,
        early_stopping_patience: Optional[int] = None,
        early_stopping_delta: float = 0.0,
        predict_square_roots: bool = False,
        profile_run: bool = False,
    ):


        # Seed rng
        self.seed = seed

        # Distributed Training variables
        self.device = device
        self.rank = rank
        self.n_devices = n_devices

        # Set model
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.predict_square_roots = predict_square_roots

        # Wrap model in DDP if distributed
        if self.n_devices > 1:
            self.model = DDP(model, device_ids=[self.rank])
        else:
            self.model = model

        self.optimizer = self.optimizer_type.init(
            self.model.parameters(), lr=lr, **optimizer_kwargs
        )

        self.loss_func = loss_func

        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps

        # Initialize data loading stuff
        self.train_path = train_path
        self.val_path = val_path

        # Build paths to tree/msa directories
        t_trees, t_msas = (
            os.path.join(self.train_path, "trees"),
            os.path.join(self.train_path, "msas"),
        )
        v_trees, v_msas = (
            os.path.join(self.val_path, "trees"),
            os.path.join(self.val_path, "msas"),
        )

        self.train_data = PhyloDataset(make_pairs(t_trees, t_msas))
        self.val_data = PhyloDataset(make_pairs(v_trees, v_msas))
        self.train_bs = train_bs
        self.val_bs = val_bs

        # Initialize distributed data smaplers
        if self.n_devices > 1:
            self.train_sampler = CheckpointableDsitributedSampler(
                self.train_data,
                batch_size=self.train_bs,
                shuffle=True,
                drop_last=False,
                seed=self.seed,
            )
            self.val_sampler = CheckpointableDsitributedSampler(
                self.val_data, batch_size=self.val_bs, shuffle=False, drop_last=True
            )
        else:
            self.train_sampler = CheckpointableRandomSampler(
                self.train_data, batch_size=self.train_bs, seed=self.seed
            )
            self.val_sampler = CheckpointableSequentialSampler(
                self.val_data, batch_size=self.val_bs
            )

        # Create dataloaders
        self.init_loaders()

        # Initialize LR scheduler
        self.scheduler_type = scheduler_type
        total_steps = (
            math.ceil(self.train_sampler.num_samples / self.train_bs) * self.n_epochs
        )
        self.scheduler = self.scheduler_type.init(
            self.optimizer, total_steps, self.warmup_steps
        )

        # Early stopping and monitoring
        self.max_steps = max_steps
        self.max_loss_threshold = max_loss_threshold
        self.check_loss_anomalies = check_loss_anomalies
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.loss_checker_warmup = loss_checker_warmup

        # Exit status variable to synchronise between processes
        self.should_shutdown = torch.zeros((1), device=self.device)
        self.exit_code = torch.zeros((1), device=self.device)
        self.shutdown_message = None

        # Makes sure there are no NaNs/infs and that the loss does not go over certain threshold
        self.loss_checker = LossChecker(
            self.max_loss_threshold, self.check_loss_anomalies, self.loss_checker_warmup
        )
        # Early stopper on val loss
        self.val_improvement_checker = EarlyStopper(
            self.early_stopping_patience, self.early_stopping_delta
        )

        self.max_step_checker = MaxStepLimiter(self.max_steps)

        # Logging
        self.wrote_headers = False
        self.project_name = project_name
        self.run_name = run_name
        self.outdir = os.path.join(self.project_name, self.run_name)
        self.artefacts = artefacts

        # Init intervals
        self.val_every = val_every
        self.log_every = log_every

        # Initialize Counters
        self.start_epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.epoch = 0

        # Initialize epoch training loss accumulator
        self.accum = self.init_accum()
        self.accum_samples = self.init_accum()

        # Additional metrics to log
        self.add_metrics = {
            "MSE": LossType.MSE,
            "MAE": LossType.MAE,
            "MRE": LossType.MRE,
        }
        self.accum_metrics = {k: self.init_accum() for k in self.add_metrics}

        # Profile instead of train
        self.profile = profile_run
        if self.profile: 
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(skip_first=10, warmup=2, active=5, wait=50, repeat=5),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.outdir, "profile"))
            )


    def to_logging_str(self) -> str:
        st = self.state_dict()

        s = ""
        # Add model characteristics
        m = st["model"]

        # Add data characteristics

        # Add trainer characteristics

        # Add outputs
        return s

    def init_loaders(self):
        self.train = DataLoader(
            self.train_data,
            batch_size=self.train_bs,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=PhyloDataset.pad_batches,
        )
        self.val = DataLoader(
            self.val_data,
            batch_size=self.val_bs,
            sampler=self.val_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=PhyloDataset.pad_batches,
        )

        # Check if we need an additional validation dataset
        # in the case where the number of validation examples
        # is not divisible by the number of devices
        self.aux_val = None
        if self.n_devices > 1:
            dist_val_samples = self.val_sampler.num_samples * self.n_devices
            total_val_samples = len(self.val_data)
            if self.rank == 0 and dist_val_samples < total_val_samples:
                aux_val_data = Subset(
                    self.val_data, range(dist_val_samples, total_val_samples)
                )
                self.aux_val = DataLoader(
                    aux_val_data,
                    batch_size=self.val_bs,
                    pin_memory=True,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=PhyloDataset.pad_batches,
                )

    def model_state(self):
        # If needed, unwrap model from DDP wrapper
        model = self.model.module if self.n_devices > 1 else self.model

        return {
            "arch": model._get_hyperparams(),
            "state": model.state_dict(),
        }

    def _trainer_config(self):
        return {
            "lr": self.lr,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "loss_func": self.loss_func,
            "n_epochs": self.n_epochs,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "train_bs": self.train_bs,
            "val_bs": self.val_bs,
            "seed": self.seed,
            "run_name": self.run_name,
            "val_every": self.val_every,
            "log_every": self.log_every,
            "project_name": self.project_name,
            "scheduler_type": self.scheduler_type,
            "warmup_steps": self.warmup_steps,
            "n_devices": self.n_devices,
            "max_steps": self.max_steps,
            "max_loss_threshold": self.max_loss_threshold,
            "check_loss_anomalies": self.check_loss_anomalies,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_delta": self.early_stopping_delta,
            "loss_checker_warmup": self.loss_checker_warmup,
            "predict_square_roots": self.predict_square_roots,
        }

    def state_dict(self):
        def receiver():
            return torch.zeros(self.n_devices, 1, device=self.device)

        if self.n_devices > 1:
            # Gather distributed epoch loss trackers
            dist_accums = receiver()
            dist_n_samples = receiver()
            dist_metrics = {k: receiver() for k in self.add_metrics}
            all_gather_into_tensor(dist_accums, self.accum)
            all_gather_into_tensor(dist_n_samples, self.accum_samples)
            for k in dist_metrics:
                all_gather_into_tensor(dist_metrics[k], self.accum_metrics[k])
        else:
            dist_accums = self.add_metrics
            dist_n_samples = self.accum_samples
            dist_metrics = self.accum_metrics

        return {
            "model": self.model_state(),
            "optimizer": self.optimizer.state_dict(),
            "trainer": self._trainer_config(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "counters": {
                "epoch": self.epoch,
                "local_step": self.local_step,
                "global_step": self.global_step,
            },
            "epoch_loss_tracker": {
                "accumulator": dist_accums,
                "n_samples": dist_n_samples,
                "metrics": dist_metrics,
            },
            "samplers": {
                "train": self.train_sampler.state_dict(),
                "val": self.val_sampler.state_dict(),
            },
            "rng": self.rng_state(),
        }

    def rng_state(self):
        rng = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda"] = torch.cuda.get_rng_state()
        elif torch.backends.mps.is_available():
            rng["torch_mps"] = torch.mps.get_rng_state()

        return rng

    def _checkpoint_path(self, step=True, epoch=True, suffix=None):
        stem = os.path.join(self.outdir, "checkpoints", self.run_name)

        if epoch:
            stem += f"-epoch-{self.epoch}"
        if step:
            stem += f"-step-{self.global_step}"
        if suffix is not None:
            stem += f"-{suffix}"

        # TODO: TEST THING (TO REMOVE)
        if self.start_epoch != 0:
            stem += "_resumed"

        return f"{stem}.ckpt"

    def save_checkpoint(
        self, path, extras=None, incr_epoch=False, incr_local=False, incr_global=False
    ):
        """
        Save a checkpoint.
        It is possible to add extra information by passing a
        dictionnary to the `extras` parameter. (Be careful with
        it since duplicate keys will overwrite the checkpoint
        information...)
        You can also increment the following variables by 1 in the saved checkpoint:
           - epoch
           - global_step
           - local_step
        This is needed so that the loaded checkpoints start at the correct step/epoch
        regardless of whether it was saved during an epoch or at the end of it.
        """

        # Needs to run on all processes since we're gathering
        # accumulators from other processes ?
        state = self.state_dict()

        if self.rank == 0:
            # Increment counters if needed
            if incr_epoch:
                state["counters"]["epoch"] += 1
            if incr_local:
                state["counters"]["local_step"] += 1
            if incr_global:
                state["counters"]["global_step"] += 1

            # Add any extra info
            if extras is not None:
                state.update(extras)

            # Write checkpoint to disk
            torch.save(state, path)

        # Make sure checkpoint is saved to disk before continuing with training
        if self.n_devices > 1:
            barrier()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def infer(self, x, y):
        """
        Compute y estimates for an input batch,
        as well as the loss value for that batch
        """

        # Make masks
        x_mask = (x[:, 0, :, :] != PADDING_TOKEN).to(self.device, non_blocking=True)
        y_mask = (y != PADDING_TOKEN).to(self.device, non_blocking=True)

        # Infer y estimate
        y_hat = self.model(x.float().to(self.device, non_blocking=True), x_mask)
        y = y.float().to(self.device, non_blocking=True)
        if self.predict_square_roots:
            y = torch.sqrt(y)

        # Compute loss + metrics
        loss = self.loss_func(y, y_hat, y_mask)

        # Square predictions if learning square roots
        m_y, m_y_hat, m_mask = y.detach(), y_hat.detach(), y_mask.detach()
        if self.predict_square_roots:
            m_y, m_y_hat = torch.pow(m_y, 2.0), torch.pow(m_y_hat, 2.0)

        metrics = {
            m_name: m_func(m_y, m_y_hat, m_mask)
            for m_name, m_func in self.add_metrics.items()
        }

        return y_hat, loss, metrics

    def validate(self):
        """
        Compute validation loss on whole validation set
        """
        # Make sure the dataloaders are initialized
        assert (
            self.val is not None
        ), "Must initialize validation loader first with `self.init_loaders()`"

        self.model.eval()
        accum, samples = (self.init_accum(), self.init_accum())
        metric_accums = {k: self.init_accum() for k in self.add_metrics}

        with (
            torch.no_grad(),
            DistributedProgressBar(
                rank=self.rank, total=len(self.val), initial=0, desc="VAL", leave=False
            ) as pbar,
        ):
            for x, y in self.val:
                # Step profiler if profiling run
                if self.prof is not None:
                    self.prof.step()

                _, loss, metrics = self.infer(x, y)

                # Accumulate loss over epoch
                # Weight average batch loss by effective batch size
                accum += loss * x.shape[0]
                samples += x.shape[0]

                # Accumulate metrics
                for m in metric_accums:
                    metric_accums[m] += metrics[m] * x.shape[0]

                pbar.update(1)

        if self.n_devices > 1:
            # Sync accumulated loss between GPUS
            all_reduce(accum, op=ReduceOp.SUM)
            all_reduce(samples, op=ReduceOp.SUM)
            for k in metric_accums:
                all_reduce(metric_accums[k], op=ReduceOp.SUM)

        # Evaluate validation loss on auxiliary validation dataset if needed
        if self.rank == 0 and self.aux_val is not None:
            for x, y in self.aux_val:
                _, loss, metrics = self.infer(x, y)
                accum += loss * x.shape[0]
                samples += x.shape[0]
                for m in metric_accums:
                    metric_accums[m] += metrics[m] * x.shape[0]

        return accum / samples, {k: v / samples for k, v in metric_accums.items()}

    def init_logger(self):
        self.wandb_logger = None
        self.logpath = ""
        if self.rank == 0:
            # Create output directories if needed
            logdir = os.path.join(self.outdir, "logs")
            ckptdir = os.path.join(self.outdir, "checkpoints")
            artfdir = os.path.join(self.outdir, "artefacts")
            for _dir in [logdir, ckptdir, artfdir]:
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)

            # Setting csv logfile path
            runname = (
                f"{self.run_name}-E{self.start_epoch}-S{self.train_sampler.start_iter}"
            )
            self.logpath = os.path.join(logdir, f"{runname}.csv")

            model = self.model.module if self.n_devices > 1 else self.model
            config = {
                "trainer": self._trainer_config(),
                "model": model._get_hyperparams(),
            }

            self.wandb_logger = wandb.init(
                name=runname,
                project=self.project_name,
                mode="offline",
                group=self.run_name,
                job_type="training",
                dir=logdir,
                config=config,
            )

            # Watch model to log gradients
            self.wandb_logger.watch(self.model)  # type: ignore

            # Log artefacts if needed
            for path, name in self.artefacts:
                # Copy file to run directory
                logged_path = os.path.join(artfdir, name)
                shutil.copyfile(path, logged_path)

                # Log to wandb
                self.wandb_logger.save(logged_path, base_path=artfdir)

    def close_logger(self, quiet: bool = False, exit_code: int = 0):
        if self.wandb_logger is not None:
            self.wandb_logger.finish(quiet=quiet, exit_code=exit_code)

    def _write_log_headers(self, logfile=None):
        self.wrote_headers = True
        if logfile is not None:
            logfile.write("epoch,global_step,local_step,value,variable\n")

    def log_var(self, varname, val, logfile=None, wandb=True):
        if not self.wrote_headers:
            self._write_log_headers(logfile)

        # CSV LOGGING
        if logfile is not None:
            logfile.write(
                f"{self.epoch},{self.global_step},{self.local_step},{val},{varname}\n"
            )

        # WANDB LOGGING
        if self.wandb_logger is not None and wandb:
            self.wandb_logger.log({varname: val})

    def log_vars(self, vardict, logfile=None):
        # CSV LOGGING
        for varname, val in vardict.items():
            self.log_var(varname, val, logfile, wandb=False)

        # WANDB LOGGING
        if self.wandb_logger is not None:
            self.wandb_logger.log(
                {"global_step": self.global_step, "epoch": self.epoch, **vardict},
            )

    def init_accum(self):
        return torch.zeros(1, device=self.device, requires_grad=False)

    def train_epochs(self):
        """
        run entire training loop
        """
        assert (
            self.train is not None
        ), "Must initialize dataloaders with `self.init_loaders()`"

        # Initialize wandb
        self.init_logger()

        def run_loss_checker(checker, loss):
            should_stop, message = checker.should_stop(loss)
            if should_stop:
                self.should_shutdown += 1.0
                self.shutdown_message = message

            # Sync stop message between GPUs if needed
            if self.n_devices > 1 and self.should_shutdown > 0:
                all_reduce(self.should_shutdown, op=ReduceOp.MAX)

        with (
            DistributedLogFile(self.logpath, self.rank, buffering=100) as logfile,
            DistributedProgressBar(
                total=self.n_epochs,
                initial=self.start_epoch,
                desc="EPOCH",
                leave=True,
                rank=self.rank,
            ) as pbar_epoch,
        ):
            accum_inner = self.init_accum()
            samples_inner = self.init_accum()
            metrics_inner = {k: self.init_accum() for k in self.add_metrics}

            for epoch in range(self.start_epoch, self.n_epochs):
                self.epoch = epoch
                self.train_sampler.set_epoch(epoch)
                self.model.train()

                # TRAINING LOOP

                with DistributedProgressBar(
                    total=len(self.train),
                    initial=self.train_sampler.start_iter,
                    desc="TRAIN",
                    leave=False,
                    rank=self.rank,
                ) as pbar_train:
                    for x, y in self.train:
                        # Step profiler if profiling run
                        if self.prof is not None:
                            self.prof.step()

                        # Check if we have reached max number of steps allowed
                        run_loss_checker(self.max_step_checker, self.global_step)
                        if self.should_shutdown > 0:
                            break  # Out of train data loop

                        # Execute a training step
                        self.optimizer.zero_grad()
                        _, loss, metrics = self.infer(x, y)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                        # Accumulate loss over epoch and inner interval
                        self.accum += (
                            loss.detach() * x.shape[0]
                        )  # Weight loss by batch size
                        self.accum_samples += x.shape[0]
                        for m in self.accum_metrics:
                            self.accum_metrics[m] += metrics[m] * x.shape[0]

                        accum_inner += loss.detach() * x.shape[0]
                        samples_inner += x.shape[0]
                        for m in metrics_inner:
                            metrics_inner[m] += metrics[m] * x.shape[0]

                        if self.global_step % self.log_every == 0:
                            # Log training loss to various loggers
                            self.log_vars(
                                {
                                    "train_inner": loss.item(),
                                    "learning_rate": self.get_lr(),
                                },
                                logfile,
                            )

                            # Check that training loss is OK
                            run_loss_checker(self.loss_checker, loss)

                        if self.should_shutdown > 0:
                            break  # Out of train data loop

                        # Measure validation loss within epoch
                        if (
                            self.global_step % self.val_every == 0
                            and self.global_step != 0
                        ):
                            # Compute average train loss from last intermediate logging,
                            # validation loss and log them
                            if self.n_devices > 1:
                                all_reduce(accum_inner, op=ReduceOp.SUM)
                                all_reduce(samples_inner, op=ReduceOp.SUM)
                                for k in metrics_inner:
                                    all_reduce(metrics_inner[k], op=ReduceOp.SUM)

                            inner_train_loss = accum_inner / samples_inner
                            inner_train_metrics = {
                                f"train_{k}": (v / samples_inner).item()
                                for k, v in metrics_inner.items()
                            }

                            # Reset inner accumulators
                            accum_inner = self.init_accum()
                            samples_inner = self.init_accum()
                            metrics_inner = {
                                k: self.init_accum() for k in self.add_metrics
                            }

                            inner_val_loss, metrics = self.validate()
                            self.log_vars(
                                {
                                    "train_loss": inner_train_loss.item(),
                                    "val_loss": inner_val_loss.item(),
                                    **inner_train_metrics,  # Additional training metrics
                                    **{
                                        f"val_{k}": v.item() for k, v in metrics.items()
                                    },  # Additional validation metrics
                                },
                                logfile,
                            )

                            # Save within-epoch checkpoint
                            self.save_checkpoint(
                                self._checkpoint_path(suffix="inner"),
                                extras={
                                    "train_loss": inner_train_loss.item(),
                                    "val_loss": inner_val_loss.item(),
                                },
                                incr_local=True,
                                incr_global=True,
                            )

                            # Check if val loss anomalous
                            for checker in [
                                self.loss_checker,
                                self.val_improvement_checker,
                            ]:
                                run_loss_checker(checker, inner_val_loss)
                                if self.should_shutdown > 0:
                                    break
                        # END OF WITHIN EPOCH VALIDATION

                        if self.should_shutdown > 0:
                            break  # Out of train data loop

                        self.global_step += 1
                        self.local_step += 1
                        pbar_train.update(1)
                    # END OF TRAIN LOOP
                # END OF TRAINING PROGRESS BAR

                if self.should_shutdown > 0:
                    break  # Out of epoch loop

                # Reset local step trackers
                self.local_step = 0
                self.train_sampler.set_starting_step(0)

                if self.n_devices > 1:
                    # Gather loss values accross GPUs
                    all_reduce(self.accum, op=ReduceOp.SUM)
                    all_reduce(self.accum_samples, op=ReduceOp.SUM)
                    for k in self.accum_metrics:
                        all_reduce(self.accum_metrics[k], op=ReduceOp.SUM)

                # Compute epoch loss values
                epoch_train_loss = (self.accum / self.accum_samples).item()
                epoch_val_loss, epoch_val_metrics = self.validate()
                epoch_train_metrics = {
                    f"train_{k}_epoch": (v / self.accum_samples).item()
                    for k, v in self.accum_metrics.items()
                }

                # Log loss values
                self.log_vars(
                    {
                        "train_loss_epoch": epoch_train_loss,
                        "val_loss_epoch": epoch_val_loss.item(),
                        **epoch_train_metrics,  # Additional train metrics
                        **{
                            f"val_{k}_epoch": v.item()
                            for k, v in epoch_val_metrics.items()
                        },  # Additional validation metrics
                    },
                    logfile,
                )

                # Reset accumulators
                self.accum = self.init_accum()
                self.accum_samples = self.init_accum()
                self.accum_metrics = {k: self.init_accum() for k in self.add_metrics}

                # Save end of epoch checkpoint
                self.save_checkpoint(
                    self._checkpoint_path(step=False, suffix="end"),
                    extras={"train_loss": epoch_train_loss, "val_loss": epoch_val_loss},
                    incr_epoch=True,
                )

                pbar_epoch.update(1)
            # END OF EPOCH LOOP
        # END OF LOGFILE AND EPOCH PROGRESS BAR

        # Synchronize exit codes between processes
        if self.n_devices > 1:
            all_reduce(self.exit_code, op=ReduceOp.MAX)
        code = int(self.exit_code.item())

        if self.prof is not None:
            self.prof.stop()

        # END OF RUN CLEANUP
        if self.should_shutdown > 0:
            # TODO: clean this garbage code up
            if self.rank == 0:
                print(
                    f"Ending Training early at epoch={self.epoch}/step={self.global_step}",
                    end="",
                )
            if self.shutdown_message is not None:
                print(
                    f" because: {self.shutdown_message}", end=""
                )  # Should send message to rank 0
            if self.rank == 0:
                print()

            if self.rank == 0:
                print("Saving checkpoint before shutting down")
            self.save_checkpoint(self._checkpoint_path(suffix="shutdown"))
        else:
            print("Finished training run successfuly.")

        # Make sure all the printing is done before closing WandB logger
        if self.n_devices > 1:
            barrier()

        self.close_logger(exit_code=code)

        return code


def trainer_from_checkpoint(path, rank, device, n_devices):
    # Load data
    state = torch.load(path, map_location={"cuda:0": f"cuda:{rank}"})

    artefacts = [(path, PurePath(path).name)]

    # Check that we are resuming training with the same amount of devices
    ckpt_devices = state["trainer"].pop("n_devices")
    assert ckpt_devices == n_devices, (
        f"Checkpoint was trained on {ckpt_devices} GPUs but "
        f"attempting to resume training on {n_devices} GPUs."
    )

    # Load model
    model = PhyloformerSeq(**state["model"]["arch"]).to(device)
    model.load_state_dict(state["model"]["state"])

    # Initialize Trainer
    # Args not in `state['trainer']`: model, device, rank, n_devices
    trainer = DistributedTrainer(
        model=model,
        device=device,
        rank=rank,
        n_devices=n_devices,
        artefacts=artefacts,
        **state["trainer"],
    )

    # Reset counters
    trainer.start_epoch = state["counters"]["epoch"]
    trainer.epoch = state["counters"]["epoch"]
    trainer.local_step = state["counters"]["local_step"]
    trainer.global_step = state["counters"]["global_step"]

    # Reset loss accumulator
    trainer.accum = state["epoch_loss_tracker"]["accumulator"][rank]
    trainer.accum_samples = state["epoch_loss_tracker"]["n_samples"][rank]
    if state["epoch_loss_tracker"].get("metrics") is not None:
        for k in trainer.accum_metrics:
            trainer.accum_metrics[k] = state["epoch_loss_tracker"]["metrics"][k][rank]

    # Reset state for certain members
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.train_sampler.load_state_dict(state["samplers"]["train"])
    trainer.train_sampler.set_starting_step(state["counters"]["local_step"])
    trainer.val_sampler.load_state_dict(state["samplers"]["val"])

    if state["scheduler"] is not None:
        assert (
            trainer.scheduler is not None
        ), "Trying to load state into inexistant scheduler"
        trainer.scheduler.load_state_dict(state["scheduler"])

    # Reset RNG state
    rng_state = state["rng"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available():
        s = rng_state["torch_cuda"] # Needed because I messed up checkpoint saving
        torch.cuda.set_rng_state(s[0] if isinstance(s, tuple) else s)
    elif torch.backends.mps.is_available():
        s = rng_state["torch_mps"] # Needed because I messed up checkpoint saving
        torch.mps.set_rng_state(s[0] if isinstance(s, tuple) else s)

    # Print start step
    if rank == 0:
        state_bkp = {
            k: deepcopy(v)
            for k, v in state.items()
            if k not in ["model", "optimizer", "rng"]
        }
        print("Restarting model from State: ")
        pprint(state_bkp)

    return trainer


@dataclass
class Updateable:
    def update(self, new: dict):
        for k, v in new.items():
            if hasattr(self, k):
                self.__setattr__(k, v)
            else:
                name = self.__class__.__name__
                raise KeyError(f"'{k}' is not a valid attribute name for {name} class.")

    def to_toml_inner(self) -> str:
        s = ""
        to_write = []
        for f in fields(self):
            if is_dataclass(getattr(self, f.name)):
                to_write.append(f.name)
                continue
            if (val := getattr(self, f.name)) is not None:
                s += f"{f.name} = {val_to_toml(val)}\n"
        for f in to_write:
            if (child := getattr(self, f)) is not None:
                s += child.to_toml()

        return s

    def to_toml(self) -> str:
        raise SystemError(f"to_toml not implemented for {self}")


@dataclass
class RunConfig:
    trainer: "TrainerConfig"
    model: "PFSeqConfig"

    @classmethod
    def from_dict(cls: Type["RunConfig"], obj: dict):
        return cls(
            trainer=TrainerConfig.from_dict(obj["trainer"]),
            model=PFSeqConfig.from_dict(obj.get("model", dict())),
        )

    def to_dict(self):
        return {
            "trainer": self.trainer.to_dict(),
            "model": self.model.to_dict(),
        }

    def to_toml(self) -> str:
        s = self.trainer.to_toml()
        s += self.model.to_toml()
        return s

    def to_trainer(
        self, rank: int, n_devices: int, device: torch.device
    ) -> DistributedTrainer:
        model = PhyloformerSeq(**asdict(self.model)).to(device)
        return DistributedTrainer(
            model=model,
            rank=rank,
            n_devices=n_devices,
            device=device,
            **asdict(self.trainer),
        )


@dataclass
class TrainerConfig(Updateable):
    # Must be provided
    train_path: str  # Path to training data
    val_path: str  # Path to validation data
    project_name: str  # Name of the project of the run
    # Default values
    train_bs: int = 8
    val_bs: int = 32
    n_epochs: int = 20
    lr: float = 1e-3
    optimizer_type: OptType = OptType.ADAM
    scheduler_type: SchedulerType = SchedulerType.CONSTANT
    loss_func: LossType = LossType.L1
    val_every: int = 1000
    log_every: int = 50
    warmup_steps: int = 0
    seed: Union[int, Optional[int]] = None
    run_name: Union[str, Optional[str]] = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None
    max_loss_threshold: Optional[float] = None
    check_loss_anomalies: bool = True
    loss_checker_warmup: int = 0
    early_stopping_patience: Optional[int] = None
    early_stopping_delta: float = 0.0
    predict_square_roots: bool = False
    artefacts: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if (
            self.train_path is None
            or self.val_path is None
            or self.project_name is None
        ):
            raise ValueError(
                "train_path, val_path and project_name must all be user-specified. "
                "One of them has been left empty or has been set to None"
            )

        if self.run_name is None:
            self.run_name = generate_run_name()
            assert self.run_name is not None
        if self.seed is None:
            self.seed = int(time())
            assert self.seed is not None

    @classmethod
    def from_dict(cls: Type["TrainerConfig"], obj: dict):
        obj = obj.copy()
        conf = cls(
            train_path=obj.pop("train_path"),
            val_path=obj.pop("val_path"),
            project_name=obj.pop("project_name"),
        )

        if (opt_s := obj.pop("optimizer_type", None)) is not None:
            conf.optimizer_type = instantiate_enum(opt_s, OptType)
        if (sched_s := obj.pop("scheduler_type", None)) is not None:
            conf.scheduler_type = instantiate_enum(sched_s, SchedulerType)
        if (loss_s := obj.pop("loss_func", None)) is not None:
            conf.loss_func = instantiate_enum(loss_s, LossType)

        conf.update(obj)

        return conf

    def to_dict(self):
        d = self.__dict__
        for k in ["scheduler_type", "optimizer_type", "loss_func"]:
            d[k] = d[k].name
        return d

    def to_toml(self):
        s = "\n[trainer]\n"
        s += self.to_toml_inner()
        return s


@dataclass
class PFSeqConfig(Updateable):
    n_blocks: int = 6
    n_heads: int = 4
    h_dim: int = 64
    dropout: float = 0.0
    use_sla: bool = False
    use_bilinear: bool = True
    use_shortcuts: bool = False
    bilinear_kwargs: Optional["BilinearFormConfig"] = None
    euclidean_kwargs: Optional["EuclideanConfig"] = None

    @classmethod
    def from_dict(cls: Type["PFSeqConfig"], obj: dict):
        obj = obj.copy()

        bil = BilinearFormConfig.from_dict(obj.pop("bilinear_kwargs", dict()))
        euc = EuclideanConfig.from_dict(obj.pop("euclidean_kwargs", dict()))
        conf = cls(bilinear_kwargs=bil, euclidean_kwargs=euc)
        conf.update(obj)

        return conf

    def to_dict(self):
        d = self.__dict__
        for k in ["bilinear_kwargs", "euclidean_kwargs"]:
            d[k] = d[k].to_dict()
        return d

    def to_toml(self):
        s = "\n[model]\n"
        s += self.to_toml_inner()
        return s


@dataclass
class EuclideanConfig(Updateable):
    eps: float = 1e-8
    use_cls: bool = False
    learnable: bool = False
    use_bias: bool = False

    @classmethod
    def from_dict(cls: Type["EuclideanConfig"], obj: dict):
        conf = cls()
        conf.update(obj)
        return conf

    def to_dict(self):
        return self.__dict__

    def to_toml(self):
        s = "\n[model.euclidean_kwargs]\n"
        s += self.to_toml_inner()
        return s


@dataclass
class BilinearFormConfig(Updateable):
    use_bias: bool = True
    use_sqrt_norm: bool = True
    use_cls: bool = True
    factorized_kernel: bool = True
    ensure_positive: bool = True
    eps: float = 1e-5

    @classmethod
    def from_dict(cls: Type["BilinearFormConfig"], obj: dict):
        conf = cls()
        conf.update(obj)
        return conf

    def to_dict(self):
        return self.__dict__

    def to_toml(self):
        s = "\n[model.bilinear_kwargs]\n"
        s += self.to_toml_inner()
        return s
