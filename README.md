# Create an AI bot playing Celeste

Celeste is a fantastic game that I have played for hours. Last year I discovered Reinforcement Learning for work issues, the idee of creating an AI using RL able to play Celeste crossed my mind in early february 2023.

##Why Celeste is intersting for a RL problem

### Celeste have really simple commands :
Maddeline, the hero of the game has only four actions :
- Moving in eight directions (left, right, up, down and diagonals)
- Jump
- Grab a wall
- Dash

There is no inventory or need to use the mouse or other actions that requires complicated mechanisms.
This advantage should allow a RL algorithm to easily learn the actions

### The format of the game is very suitable to a Reinforcement Learning problem.
If you have never played Celeste, the game is devided into seven levels (for the main game), but each level is also devided into several screen.
In each screen Maddeline has a respawn point, the goal is to go from the start of the screen to the end without dying.
"Screen to insert"
If you die, no problem you just respawn where you where without any penalties. This create very good episodics and statics environnement, you can die hundreads of time there is no game over, only your death counter will increase a lot.

Because their is several screens but all static, you can in fact create a changing environnement but also static, that is really intersting for RL algorithms.


### Celeste is emulatable
One really import part of a Reinforcement Learning problem is how quickly the environnement will execute and how you can interact with the environnement.
Well thanks to the Celeste Community, all this is possible. 
I am not at all a super great game developper, Celeste is a c# game and all RL algorithm used python librairies. Interaction between Python and C# would have been to hard for me to make. 

But Celeste has a Great community that create mods and espacially one, CelesteTAS. A TAS is a bot playing Celeste to which you provide all the commands at each actions, it allow players to test the limits of the game. This mod work with a text editor that send request to the mod and allow several things: 
- Send actions to the game
- Load screen where you want in a level
- Accelerate or Pause the game
- Get current informations like the position, speed etc of Maddelin

Thanks to CelesteTAS I have been able to create an easy iteraction with the game that make the game around 3 times quicker that the normal game and with more accuracy.


## Reinforcement Learning problem formalisation

An episode start with Maddeline, and for know it end either if Maddeline died or change screen (or at a certain number of steps). In the future I will delete the end of an episode if Maddeline change screen.

To create a Reinforcement Learning problem, you have to identity several components:
### The Actor
Obviously the Actor is Maddeline

### The actions

An important point here : Celeste is a 60 fps game, it mean that you can litteraly take 60 actions per step, which is A LOT ! I decided to simplify that by giving 5 frames per actions, it make the learning easier and for several reasons it simplifing other things (some action do not really apply if you do them 1 frame)

I listed above the different actions ingame, it is quite the same of the RL problem but with two distinction : 
- Maddeline do not have one action to move in eight direction, but two actions two move in two directions, one to go right and left and one to go up and down
- If Maddeline jump for only one frame, it make a little jump, so I decide to devide the jump action into three : no jump, jump one frame, jump five frames

This lead to this multi-action space [3, 3, 2, 3, 2]
 - Action for moving horizontaly : 0 for left, 1 for nothing, 2 for right
 - Action for moving vertically : 0 for down, 1 for nothing, 2 for up
 - Action for Dashing : 0 for nothing, 1 for dashing
 - Action for Jumping : 0 for no jump, 1 for little jump, 2 for big jump
 - Action for Grab/Climb : 0 for no grab, 1 for grab

### The observation space

#### First case : only the information
With all the data given by the CelesteTAS mod, a observation is possible, the standard observation size is eleven with value between 0 and 1 or -1 and 1 :
- 0, min=0, max=1 : Position on x axis noramlize with the size of the screen
- 1, min=0, max=1: Position on y axis normalize with the size of the screen
- 2, min=-1, max=1 : Speed on x reduce by 6
- 3, min=-1, max=1 : Speed on y reduce by 6
- 4, min=0, max=1 : Stamina, max stamina is 110 so stamina is devide by 110
- 5, min=0, max=1 : 0.5 if maddeline touch a wall on the left, 1 if maddeline touch a wall on the right, else 0 
- 6, min=0, max=1 : State of maddeline, 0.5 if she is dashing, 1 if she is climb, else 0
- 7, min=0, max=1 : 1 If Maddeline can dash, else 0
- 8, min=0, max=1 : Particular value due to Celeste, normaly you should only can jump if you touch the ground, but Celeste is a generous game, if you leave the ground you still have 5 frames to jump. So Value here is 1 if maddeline is on the ground, else the number of frame until maddeline can not jump (divide by 5) and 0 if Maddeline can not jump
- 9, min=0, max=1 : When Maddeline is jumping, there is an info about how much step there are until jump action end. It is also normalize to one
- 10, min=0, max=1 : When Maddeline is dashing, there is an info about how much step there are until the dash end. It is also normalize to one

with this some info can be added like the coordonates of the goal of the screen or the index of the screen. Has well has a historic information with the states of the former states

#### Second case : The image game
There is a possibility to give the whole screen image to Maddeline with a reduction factor. It should allow the agent to really understand a screen and to learn how to finished a screen that have not been trained before.

You can has well give historic of image.

With the image case, you still give all the information in first case, there is a CNN to use the image then concatenate with general informations.

### The reward

pijapijpz

## What algorithms are implemented

For me, this project was a great opportunity to implement and test myself several algorithms

### Deep Q-Learning
I first start with an easy Deep Reinforcement Learning algorithm just to test that the environnement work fine.

Nothing really interesting
