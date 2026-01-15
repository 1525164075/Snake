# RL Snake 实时播放设计

## 目标
- 支持基于已训练好的 `checkpoint` 进行实时窗口播放，用于论文展示和直观评估策略表现。

## 范围
- 新增 `--mode play` 的演示入口。
- 复用现有环境渲染逻辑（`SnakeEnv.render(mode="human")`）。
- 不改动训练/评估/GA 逻辑。

## 总体方案
在 `scripts/train.py` 中新增 `play` 分支。用户提供 `checkpoint`，系统读取配置并构建环境与网络，加载权重后进入实时播放循环：`reset -> select_action -> step -> render`。每步处理 `pygame` 事件以支持关闭窗口退出，并用 `fps` 控制播放速度。

## CLI 参数设计
- `--mode play`：进入实时播放模式。
- `--checkpoint <path>`：必填，模型权重路径。
- `--fps <int>`：默认 10，帧率控制；为 0 时不 sleep（最快播放）。
- `--episodes <int>`：默认 3，播放回合数。
- `--max-steps <int>`：可选，覆盖配置里的 `env.max_steps`。
- `--scale <int>`：可选，渲染放大倍数（传给 `render`）。

## 关键流程
1. 解析参数并加载配置。
2. 按配置或覆盖值构建环境。
3. 创建 `QNetwork` + `DQNAgent`，加载 `checkpoint` 权重并 `update_target()`。
4. 逐回合播放：`reset -> while not done -> select_action -> step -> render(human)`。
5. 每步处理 `pygame.event.get()`，遇到 `QUIT` 直接退出。

## 错误处理
- `checkpoint` 缺失或路径不存在：输出提示并返回非 0。
- `pygame` 未安装：提示 `pip install pygame` 并退出。
- `fps <= 0`：不 sleep，避免除零。

## 测试与验收
- 新增 CLI 解析测试：`--mode play` 能被识别。
- 手动验收：使用训练得到的 `model.pt` 播放 1~3 回合，窗口正常渲染且可关闭退出。

## 示例命令
```bash
PYTHONPATH=src python3 scripts/train.py --mode play \
  --checkpoint runs/train_xxx/model.pt --fps 10 --episodes 3
```
