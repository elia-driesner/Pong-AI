import pygame
from pong import Game
import neat
import os


class PongGame():
    def __init__(self):
        self.game = Game()
        self.paddle_left = self.game.paddle_left
        self.paddle_right = self.game.paddle_right
        self.ball = self.game.ball
        self.run = True
        
    def playerVsAi():
        pong.loop()
        
    def playerVsPlayer(self):
        while self.run:
            self.game.loop()
            self.game.player_ai_move(0)
            
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-00')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
            
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    
    pong = PongGame()
    pong.playerVsPlayer()
