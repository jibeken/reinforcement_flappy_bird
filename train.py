import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import random
from collections import defaultdict

WIDTH = 400
HEIGHT = 500
GRAVITY = 2.0
JUMP_VEL = -7
PIPE_GAP = 160
PIPE_SPEED = 3
PIPE_W = 55
BIRD_X = 75
BIRD_H = 30

PIPE_SEQUENCE = [150, 220, 100, 260, 180, 120, 240, 160, 200, 140, 210, 170, 130, 250, 190]

class FlappyEnv:
    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0.0
        self.pipe_x = WIDTH + 50
        self.pipe_idx = 0
        self.pipe_top = PIPE_SEQUENCE[0]
        self.score = 0
        self.done = False
        return self._state()

    def _state(self):
        gap_center = self.pipe_top + PIPE_GAP // 2
        rel_y = int(np.clip((self.bird_y - gap_center + 250) / 50, 0, 9))
        dist = int(np.clip((self.pipe_x - BIRD_X) / 50, 0, 8))
        vel = int(np.clip((self.bird_vel + 12) / 4, 0, 5))
        return (rel_y, dist, vel)

    def step(self, action):
        if action == 1:
            self.bird_vel = JUMP_VEL
        self.bird_vel = min(self.bird_vel + GRAVITY, 12)
        self.bird_y += self.bird_vel
        self.pipe_x -= PIPE_SPEED

        if self.pipe_x < -PIPE_W:
            self.score += 1
            if self.pipe_idx == len(PIPE_SEQUENCE) - 1:
                self.done = True
                self.pipe_x = WIDTH + 500
            else:
                self.pipe_x = WIDTH + 50
                self.pipe_idx = (self.pipe_idx + 1) % len(PIPE_SEQUENCE)
                self.pipe_top = PIPE_SEQUENCE[self.pipe_idx]

        if self.bird_y < 0:
            self.bird_y   = 0
            self.bird_vel = 0

        reward = 0
        pipe_bot = self.pipe_top + PIPE_GAP
        in_x = self.pipe_x < BIRD_X + BIRD_H and self.pipe_x + PIPE_W > BIRD_X

        if in_x and (self.bird_y < self.pipe_top or self.bird_y + BIRD_H > pipe_bot):
            reward = -20; self.done = True
        elif self.bird_y + BIRD_H > HEIGHT - 40:
            reward = -20; self.done = True

        if self.pipe_x + PIPE_W < BIRD_X and self.pipe_x + PIPE_W + PIPE_SPEED >= BIRD_X:
            reward += 10

        return self._state(), reward, self.done

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.Q = defaultdict(lambda: [0.0, 0.0])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return 1 if random.random() < 0.25 else 0
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def draw_game(ax, env, episode, ep_reward, epsilon):
    ax.cla()
    ax.set_xlim(0, WIDTH); ax.set_ylim(HEIGHT, 0)
    ax.set_facecolor('#87CEEB'); ax.axis('off')
    ax.set_title(f'Episode {episode}  |  epsilon = {epsilon:.3f}', fontsize=11)

    ax.add_patch(patches.Rectangle((0, HEIGHT - 40), WIDTH, 40, color='#5F5E5A', zorder=3))
    ax.add_patch(patches.Rectangle((0, HEIGHT - 48), WIDTH, 10, color='#3B6D11', zorder=3))

    px, pt = env.pipe_x, env.pipe_top
    ax.add_patch(patches.Rectangle((px, 0), PIPE_W, pt, color='#1D9E75', zorder=2))
    ax.add_patch(patches.Rectangle((px, pt + PIPE_GAP), PIPE_W, HEIGHT - pt - PIPE_GAP, color='#1D9E75', zorder=2))
    ax.add_patch(patches.Rectangle((px - 5, pt - 18), PIPE_W + 10, 18, color='#0F6E56', zorder=2))
    ax.add_patch(patches.Rectangle((px - 5, pt + PIPE_GAP), PIPE_W + 10, 18, color='#0F6E56', zorder=2))

    bx, by = BIRD_X, env.bird_y
    ax.add_patch(patches.Circle  ((bx+15, by+15), 15, color='#EF9F27', zorder=4))
    ax.add_patch(patches.Ellipse ((bx+7,  by+18), 14, 9, color='#BA7517', zorder=4))
    ax.add_patch(patches.Circle  ((bx+20, by+10), 4, color='white', zorder=5))
    ax.add_patch(patches.Circle  ((bx+21, by+10), 2, color='black', zorder=6))
    ax.add_patch(patches.Polygon ([[bx+28, by+13],[bx+38, by+16],[bx+28, by+19]], color='#993C1D', zorder=5))

    ax.text(8, 25, f'Score:  {env.score}', fontsize=10, color='black', fontweight='bold', zorder=7)
    ax.text(8, 48, f'Reward: {ep_reward:.0f}', fontsize=10, color='black', fontweight='bold', zorder=7)

def train(episodes=600):
    env = FlappyEnv()
    agent = QLearningAgent()

    ep_rewards = []
    accum = []

    fig, (ax_game, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#f5f5f5')
    ax_plot.set_xlabel('Episodes', fontsize=12)
    ax_plot.set_ylabel('Accumulated Reward', fontsize=12)
    ax_plot.set_title('Q-Learning: Training Progress', fontsize=13, fontweight='bold')
    ax_plot.set_ylim(-25, 160)
    ax_plot.grid(True, alpha=0.3)
    line_raw, = ax_plot.plot([], [], color='#D85A30', linewidth=1, alpha=0.3, label='за эпизод')
    line_avg, = ax_plot.plot([], [], color='#1D9E75', linewidth=2, label='среднее (100 эп)')
    ax_plot.legend(fontsize=9)
    plt.tight_layout()
    plt.ion()
    plt.show()

    print('Training started...')

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0
        steps = 0

        while not env.done and steps < 2600:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps += 1

            if steps % 15 == 0:
                draw_game(ax_game, env, ep, ep_reward, agent.epsilon)
                plt.pause(0.001)

        agent.decay_epsilon()
        ep_rewards.append(ep_reward)

        window = min(100, len(ep_rewards))
        avg = np.mean(ep_rewards[-window:])
        accum.append(avg)

        line_raw.set_data(range(1, ep + 1), ep_rewards)
        line_avg.set_data(range(1, ep + 1), accum)
        ax_plot.set_xlim(1, max(10, ep))
        ax_plot.relim()
        ax_plot.autoscale_view(scaley=False)
        plt.pause(0.001)

        if ep % 100 == 0:
            avg100 = np.mean(ep_rewards[-100:])
            print(f'  ep {ep:4d} | avg reward: {avg100:7.1f} | epsilon: {agent.epsilon:.3f} | Q-states: {len(agent.Q)}')

    plt.ioff()
    print('Training complete!')
    return agent, ep_rewards, accum

def save_plot(ep_rewards, accum):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(ep_rewards) + 1), ep_rewards, color='#D85A30', linewidth=1, alpha=0.3, label='за эпизод')
    ax.plot(range(1, len(accum) + 1), accum, color='#1D9E75', linewidth=2, label='среднее (100 эп)')
    ax.set_ylim(-25, 160)
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Accumulated Reward', fontsize=12)
    ax.set_title('Q-Learning: Flappy Bird Training Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    final_avg = np.mean(accum[-50:])
    ax.axhline(y=final_avg, color='#185FA5', linewidth=1.5, linestyle='--', label=f'Plateau ≈ {final_avg:.0f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('reward_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Plot saved: reward_plot.png')

def make_gif(agent, filename='flappy_trained.gif', num_episodes=3):
    play = QLearningAgent(epsilon=0.0)
    play.Q = agent.Q
    env = FlappyEnv()
    frames = []

    for ep in range(num_episodes):
        state = env.reset()
        steps = 0
        while not env.done and steps < 2000:
            action = play.choose_action(state)
            state, _, done = env.step(action)
            frames.append({
                'by': env.bird_y, 'px': env.pipe_x,
                'pt': env.pipe_top, 'score': env.score, 'ep': ep + 1
            })
            if done:
                break
            steps += 1

    print(f'Recording {len(frames)} frames across {num_episodes} episodes...')

    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('#87CEEB')

    def draw(i):
        f = frames[i]
        ax.cla(); ax.set_xlim(0, WIDTH); ax.set_ylim(HEIGHT, 0)
        ax.set_facecolor('#87CEEB'); ax.axis('off')
        ax.set_title(f"Trained agent — ep {f['ep']}", fontsize=10)

        ax.add_patch(patches.Rectangle((0, HEIGHT-40), WIDTH, 40, color='#5F5E5A', zorder=3))
        ax.add_patch(patches.Rectangle((0, HEIGHT-48), WIDTH, 10, color='#3B6D11', zorder=3))

        px, pt = f['px'], f['pt']
        ax.add_patch(patches.Rectangle((px, 0), PIPE_W, pt, color='#1D9E75', zorder=2))
        ax.add_patch(patches.Rectangle((px, pt+PIPE_GAP), PIPE_W, HEIGHT-pt-PIPE_GAP, color='#1D9E75', zorder=2))
        ax.add_patch(patches.Rectangle((px-5, pt-18), PIPE_W+10, 18, color='#0F6E56', zorder=2))
        ax.add_patch(patches.Rectangle((px-5, pt+PIPE_GAP), PIPE_W+10, 18, color='#0F6E56', zorder=2))

        bx, by = BIRD_X, f['by']
        ax.add_patch(patches.Circle  ((bx+15, by+15), 15, color='#EF9F27', zorder=4))
        ax.add_patch(patches.Ellipse ((bx+7,  by+18), 14, 9, color='#BA7517', zorder=4))
        ax.add_patch(patches.Circle  ((bx+20, by+10), 4, color='white', zorder=5))
        ax.add_patch(patches.Circle  ((bx+21, by+10), 2, color='black', zorder=6))
        ax.add_patch(patches.Polygon ([[bx+28,by+13],[bx+38,by+16],[bx+28,by+19]], color='#993C1D', zorder=5))

        ax.text(10, 28, f"Score: {f['score']}", fontsize=13, color='black', fontweight='bold', zorder=7)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=40)
    anim.save(filename, writer=PillowWriter(fps=25))
    plt.close()
    print(f'GIF saved: {filename}')

if __name__ == '__main__':
    agent, ep_rewards, accum = train(episodes=600)
    save_plot(ep_rewards, accum)
    make_gif(agent, filename='flappy_trained.gif', num_episodes=3)