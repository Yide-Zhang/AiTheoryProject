# AI3603-Billiards
AI3603课程台球大作业

## 依赖安装

- 使用 `pip install pooltool-billiards --extra-index-url https://archive.panda3d.org/` 安装 `pooltool`.

- 使用 `pip install bayesian-optimization` 安装 `bayes_opt`.

- `Yide-Zhang` 使用的 Python 版本为 3.12.

## 代码可读性

- 确保代码具有可读性, 并附上必要的注释

## 关键文件说明

| 文件 | 作用 | 在最终测试中是否可修改 |
|------|------|-----------|
| `poolenv.py` | 台球环境（游戏规则） | ❌ 不可修改 |
| `agent.py` | Agent 定义（在 `NewAgent` 中实现你的算法） | ✅ 可修改 `NewAgent` |
| `evaluate.py` | 评估脚本（运行对战） | ✅ 可修改 `agent_b` |
| `PROJECT_GUIDE.md` | 项目详细指南 | 📖 参考文档 |
| `GAME_RULES.md` | 游戏规则说明 | 📖 参考文档 |

---