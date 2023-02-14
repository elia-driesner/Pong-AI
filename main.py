import pygame
from pong import Game
import neat
import os
import random


class PongGame():
    def __init__(self, window):
        self.window = window
        self.game = Game(window)
        self.paddle_left = self.game.paddle_left
        self.paddle_right = self.game.paddle_right
        self.ball = self.game.ball
        self.run = True
        
    def test_ai(self):
        self.game.loop()
        self.game.draw()
        # move ai
        
    def playerVsPlayer(self):
        while self.run:
            self.game.loop()
            self.game.move()
            self.game.draw()
    
    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        
        
        while self.run:
            
            output1 = net1.activate((self.game.paddle_left.y, self.game.ball.y, abs(self.game.paddle_left.x - self.game.ball.x)))
            decision1 = output1.index(max(output1))
            output2 = net2.activate((self.game.paddle_right.y, self.game.ball.y, abs(self.game.paddle_right.x - self.game.ball.x)))
            decision2 = output2.index(max(output2))
            print(decision1, decision2)
            self.game.ai_move(decision1, decision2)
            
            game_info = self.game.loop()
            self.game.draw('hits')
            
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50 or game_info.right_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
            
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

def eval_genomes(genomes, config):
    WINDOW_SIZE = [1100, 700]
    window = pygame.display.set_mode(WINDOW_SIZE)
    
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window)
            game.train_ai(genome1, genome2, config)
        
        
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-00')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    winner = p.run(eval_genomes, 50)
            
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    run_neat(config)
    
    # pong = PongGame(pygame.display.set_mode([1100, 700]))
    # pong.playerVsPlayer()
