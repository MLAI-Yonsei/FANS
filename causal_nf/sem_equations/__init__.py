from .chain import Chain
from .chain_4 import Chain4
from .chain_5 import Chain5
from .triangle import Triangle
from .collider import Collider
from .fork import Fork
from .diamond import Diamond
from .simpson import Simpson
from .simpson2 import Simpson2
from .large_backdoor import LargeBackdoor
from .german_credit import GermanCredit
from .fork2 import Fork2
from .fork3 import Fork3
from .fork4 import Fork4
from .fork5 import Fork5
from .fork6 import Fork6
from .fork3_del import Fork3Del
from .node10 import Node10


sem_dict = {}

sem_dict["chain"] = Chain
sem_dict["chain-4"] = Chain4
sem_dict["chain-5"] = Chain5
sem_dict["triangle"] = Triangle
sem_dict["collider"] = Collider
sem_dict["fork"] = Fork
sem_dict["diamond"] = Diamond
sem_dict["simpson"] = Simpson
sem_dict["simpson2"] = Simpson2
sem_dict["large-backdoor"] = LargeBackdoor
sem_dict["german"] = GermanCredit
sem_dict["fork2"] = Fork2
sem_dict["fork3"] = Fork3
sem_dict["fork4"] = Fork4
sem_dict["fork5"] = Fork5
sem_dict["fork6"] = Fork6
sem_dict["fork3_del"] = Fork3Del
sem_dict["node10"] = Node10

