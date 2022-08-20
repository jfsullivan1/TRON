# TRON
A command-line version of the game TRON. The only portion of this game coded by me was the AI for the player bot, which wins 9/10 games on average on any difficulty. The rest was supplied by Brown University. (i.e., the maps, game code itself, etc.)
This version of TRON is two-player, alternating-move, and zero-sum.
Players will move up-down-left-right leaving behind an unpassable barrier.
The first player to touch one of these barriers loses. 
The "player" bot (us) is coded using Alpha-Beta pruning with a cutoff, so only so many branches are searched before making a decision. I found this to be the most efficient at solving this issue by using Reinforcement Learning. You can play against different difficulties of bots. 

## USAGE
The game can be started from the command line in the directory of this games file location, by using:
```bash
python gamerunner.py
```

## COMMANDS
Choose the bots that will play against eachother with this flag
```bash
-bots <bot1> <bot2>
```
### IMPORTANT: 
To play with the AI bot I coded, be sure to use StudentBot as one of your bots.

Maps can be found in the maps folder. Choose the map with:
```bash
-map <path to map>
```

### Opponents:
RandBot chooses moves at random.
WallBot hugs walls as closely as possible.
TA-Bot1 and TA-Bot2 are difficult opponents, with unknown implementations of how they work. 
