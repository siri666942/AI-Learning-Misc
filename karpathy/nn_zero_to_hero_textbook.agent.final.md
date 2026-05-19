# Karpathy《Neural Networks: Zero to Hero》中文教材

> **教材说明**：本教材是 Andrej Karpathy 经典视频课程 [*Neural Networks: Zero to Hero*](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) 的完整中文学习 companion，不是提纲，不是划重点，而是对视频中理解性讲解的最大程度忠实还原。每章穿插原始代码，并在章末附上 Karpathy 推荐的延伸阅读与练习资源。

---

## 关于本教材

### 定位与使用方式

本教材定位为**配合 Karpathy 视频课程的中文详解教材**。它不是为了替代视频，而是为了在你复习时提供一个结构化的文字参考——当你想要快速回顾某个概念的讲解、查看某段代码的实现、或者寻找课后练习资源时，这份教材应该成为你的首选。

建议的使用方式是：
1. **先看视频** —— 感受 Karpathy 的讲解节奏和直觉构建过程
2. **再读教材对应章节** —— 用文字梳理和巩固理解
3. **动手跑代码** —— 教材中保留了完整的可运行代码
4. **做课后练习** —— 每章末尾附有 Karpathy 推荐的练习和拓展资源

### 前置知识

Karpathy 的这个系列从"零基础"开始，但 assumes 你有：
- **Python 基础** —— 能读写基本的 Python 代码
- **高中微积分的模糊记忆** —— 知道导数是什么概念即可，不需要熟练计算
- **好奇心** —— 这是最重要的

### 学习路径概览

| 章节 | 内容 | 核心技能 |
|------|------|----------|
| 第1章 | micrograd — 反向传播 | 从零实现自动微分引擎 |
| 第2章 | makemore Part 1 — Bigram 语言模型 | 语言建模基础、torch.Tensor |
| 第3章 | makemore Part 2 — MLP | Embedding、多层感知机、训练基础 |
| 第4章 | makemore Part 3 — Activations & BatchNorm | 深度网络训练诊断、BatchNorm |
| 第5章 | makemore Part 4 — Backprop Ninja | 手动反向传播、梯度流理解 |
| 第6章 | makemore Part 5 — WaveNet | 层次化架构、torch.nn |
| 第7章 | Let's build GPT — Transformer | Self-Attention、GPT 架构 |
| 第8章 | GPT Tokenizer — 分词器 | BPE 算法、Tokenizer 实现 |
| 第9章 | State of GPT — 全景图 | 训练流程、应用架构 |
| 第10章 | Reproduce GPT-2 — 复现 | 分布式训练、工程实践 |

### 语言说明

本教材以**中文为主**，所有概念讲解和直觉阐释使用中文。关键技术术语在首次出现时保留英文并附中文解释，例如："**反向传播**（backpropagation）"或"**嵌入**（embedding）"。代码中的变量名、注释保留英文，但在代码块前后提供中文解读。

---



---

# 第1课：micrograd — 神经网络与反向传播的彻底拆解

> *"What I cannot create, I do not understand." — Richard Feynman*

---

## 1.1 导论：为什么从微分开始

### 1.1.1 神经网络的训练本质

当我们谈论训练一个神经网络时，本质上是在解决一个优化问题：我们有一组带有参数（parameters）的数学表达式，这些参数决定了网络的输入-输出映射关系，而我们的目标是找到一组参数值，使得网络在给定任务上的表现尽可能好。如何量化"表现好坏"？通过**损失函数（loss function）**。损失函数接收网络的预测输出和真实标签，返回一个标量值——损失值越小，说明预测越接近真实。因此，训练神经网络就等价于**调整参数使损失函数的值尽可能小**。

但参数空间往往是高维且复杂的，我们无法通过枚举来找到最优解。设想一个拥有数百万参数的神经网络，每个参数都可以取无穷多个实数值，盲目搜索无异于大海捞针。我们需要一种系统性的方法，告诉每个参数应该朝哪个方向调整、调整多少。这就是**梯度（gradient）**的核心作用。

梯度是一个向量，指向函数值增长最快的方向。相反，沿梯度的反方向移动，函数值下降最快。这个看似简单的观察构成了整个深度学习训练的基石：**计算损失函数对每个参数的梯度，然后让每个参数沿梯度反方向迈出一小步**。重复这个过程，损失值便会逐步下降，网络的预测能力也随之提升。这个方法被称为**梯度下降（gradient descent）**。

那么，如何计算损失函数对每个参数的梯度呢？对于简单的线性模型，我们可以手动推导出梯度的解析表达式。但对于深层神经网络，参数之间通过层层复合函数相互耦合，手动推导会变得极其繁琐。我们需要一种自动化的机制来计算任意复杂表达式的梯度——这就是**自动微分（automatic differentiation）** engine 的使命，而 **micrograd** 正是这样一个极简但完整的自动微分引擎。

### 1.1.2 标量自动微分引擎的核心思想

micrograd 的代码量仅有约150行Python，但它蕴含了训练所有现代深度神经网络所需的核心机制。其设计思想可以概括为一句话：**构建计算图（computation graph），反向应用链式法则（chain rule）**。

具体来说，当我们写下一串数学运算时，micrograd 并不只是计算最终的数值结果，而是在背后悄悄构建一张有向无环图（DAG, Directed Acyclic Graph）。图中的每个节点是一个标量值，每条有向边代表一个运算依赖关系。例如，表达式 `d = (a + b) * c` 会被表示为：先有一个加法节点 `e = a + b`，然后有一个乘法节点 `d = e * c`。`d` 依赖于 `e`，`e` 又依赖于 `a` 和 `b`。

这张计算图完整地记录了从输入到输出的每一个计算步骤。一旦我们拥有了这张图，就可以从输出节点出发，沿着边反向遍历，逐步计算每个节点对最终输出的梯度。这种"从输出反向传播梯度到输入"的算法，就是著名的**反向传播（backpropagation）**算法，而它之所以能够工作，完全依赖于微积分中的**链式法则**。

micrograd 刻意选择了标量（scalar）作为基本操作单位——也就是说，它不做矩阵运算，每个乘法都是两个数的乘法，每个加法都是两个数的加法。这种设计让它极尽简洁，但功能却丝毫不打折扣：只要将神经网络中的每个神经元拆解为独立的加法和乘法操作，micrograd 就足以构建和训练完整的深度神经网络来完成二分类任务。理解 micrograd 的工作原理，就等于理解了 PyTorch、TensorFlow 等主流框架中 autograd 系统的底层逻辑。

---

## 1.2 导数与链式法则

在深入到代码实现之前，我们必须先打好数学基础。这一节的目标不是让你成为微积分专家，而是建立对导数和链式法则的直觉理解——这种直觉将贯穿整个深度学习的学习旅程。

### 1.2.1 导数的直觉理解

**导数（derivative）**回答的核心问题是：当我稍微扰动函数的输入时，输出会怎么变化？形式化地说，对于单变量函数 $f(x)$，其在点 $x$ 处的导数定义为：

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

这个定义说的是：让输入 $x$ 产生一个极小的增量 $h$，观察输出的变化量 $f(x+h) - f(x)$，两者的比值就是"变化率"。当 $h$ 趋近于0时，这个比值趋近于一个极限值，即导数。

导数有三种典型情况，理解它们对调试神经网络至关重要：

- **导数为正**（$f'(x) > 0$）：增大输入会使输出增大。这意味着如果我想要增加输出，应该增大输入；反之，如果想减小输出，应该减小输入。
- **导数为负**（$f'(x) < 0$）：增大输入会使输出减小。这与正导数的情况恰好相反。
- **导数为零**（$f'(x) = 0$）：输入的微小变化几乎不影响输出——这一点处于函数的局部极值（峰值或谷底）或鞍点。

在神经网络训练中，导数的符号直接告诉我们参数应该朝哪个方向调整。如果一个参数的梯度为正，说明增大这个参数会使损失增加，因此我们应该减小它（沿着梯度的反方向移动）。

由于我们大多数时候无法精确计算极限，实践中使用**数值导数（numerical derivative）**来近似：选择一个非常小的 $h$（比如 $10^{-6}$），直接计算 $[f(x+h) - f(x)] / h$。这种方法被称为**前向差分（forward difference）**。更精确的做法是**对称导数（symmetric derivative）**，即用 $[f(x+h) - f(x-h)] / 2h$ 来近似，这种方法的截断误差更小。在下面的代码中，我们用 Python 来验证数值导数与解析导数的一致性：

```python
import math

def f(x):
    """一个简单的多项式函数: 3x^2 - 4x + 5"""
    return 3 * x ** 2 - 4 * x + 5

def df_analytical(x):
    """f(x) 的解析导数: 6x - 4"""
    return 6 * x - 4

def df_numerical(x, h=1e-6):
    """用前向差分近似导数"""
    return (f(x + h) - f(x)) / h

def df_symmetric(x, h=1e-6):
    """用对称导数近似导数（更精确）"""
    return (f(x + h) - f(x - h)) / (2 * h)

x = 3.0
print(f"f(x)          = {f(x)}")
print(f"解析导数       = {df_analytical(x)}")
print(f"数值导数(前向) = {df_numerical(x)}")
print(f"数值导数(对称) = {df_symmetric(x)}")
```

运行这段代码，你会看到解析导数 `6*3 - 4 = 14` 与数值近似的结果非常接近（差异通常在 $10^{-6}$ 量级）。这种数值检验方法——用微小扰动来验证梯度计算的正确性——在实现复杂的自动微分系统时是必不可少的调试工具。Karpathy 将其称为**梯度检查（gradient checking）**，是每一个从零搭建反向传播系统的人都应该掌握的技能。

### 1.2.2 链式法则的直觉

现在我们来理解反向传播赖以生存的数学基础——**链式法则**。链式法则处理的是**复合函数**的求导问题。假设有一个函数 $z = f(y)$，而 $y = g(x)$，那么 $z$ 对 $x$ 的导数等于 $z$ 对 $y$ 的导数乘以 $y$ 对 $x$ 的导数：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

这个公式的美在于它的**可组合性**。想象一条链条：$x$ 影响 $y$，$y$ 影响 $z$。如果我想知道 $x$ 对 $z$ 的影响有多大，只需要将链条上每一段的局部影响相乘即可。$dz/dy$ 告诉我们 $y$ 变化一点点时 $z$ 变多少，$dy/dx$ 告诉我们 $x$ 变化一点点时 $y$ 变多少，两者的乘积自然就是 $x$ 对 $z$ 的总影响。

这个直觉可以自然推广到更长的链条。如果 $a \to b \to c \to d$，那么：

$$\frac{dd}{da} = \frac{dd}{dc} \cdot \frac{dc}{db} \cdot \frac{db}{da}$$

更重要的是，链式法则也适用于**分叉**的情况。如果 $a$ 同时影响 $b$ 和 $c$，而 $b$ 和 $c$ 都影响 $d$，那么 $a$ 对 $d$ 的总影响是两条路径贡献的**和**：

$$\frac{\partial d}{\partial a} = \frac{\partial d}{\partial b} \cdot \frac{\partial b}{\partial a} + \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a}$$

这里的符号从 $d$ 变成了 $\partial$，因为我们处理的是多变量函数——这种"多条路径贡献累加"的特性，是反向传播实现中梯度必须**累加**而非**赋值**的根本原因。我们稍后会在代码中看到这一点的重要性。

Karpathy 用了一个生动的比喻：梯度像水一样从输出端流向输入端。输出节点的梯度为1（毕竟"某个量对自身的变化率"就是1），然后这"梯度之流"流经计算图的每条边。每经过一个节点，流量就按照该节点的局部导数进行缩放。最终在输入端汇聚的流量，就是损失函数对该输入参数的梯度。

### 1.2.3 数值验证：一个多变量复合函数

让我们用一个稍微复杂一点的函数来验证链式法则在数值上的正确性。考虑函数 $f(a, b, c) = -a^3 + \sin(3b) - 1/c + b^{2.5} - a^{0.5}$，我们想计算它在某一点对 $a, b, c$ 的偏导数。

```python
import math

def f(a, b, c):
    """多变量复合函数"""
    return -a**3 + math.sin(3*b) - 1.0/c + b**2.5 - a**0.5

# 在点 (a=2.0, b=3.0, c=4.0) 处计算偏导数
a, b, c = 2.0, 3.0, 4.0
h = 1e-6

# 对 a 的偏导数: 保持 b, c 不变, 只扰动 a
df_da = (f(a + h, b, c) - f(a - h, b, c)) / (2 * h)
# 解析结果: -3*a^2 - 0.5*a^(-0.5) = -3*4 - 0.5/sqrt(2) = -12 - 0.3535...
df_da_analytical = -3 * a**2 - 0.5 * a**(-0.5)

# 对 b 的偏导数: 3*cos(3b) + 2.5*b^1.5
df_db = (f(a, b + h, c) - f(a, b - h, c)) / (2 * h)
df_db_analytical = 3 * math.cos(3*b) + 2.5 * b**1.5

# 对 c 的偏导数: 1/c^2
df_dc = (f(a, b, c + h) - f(a, b, c - h)) / (2 * h)
df_dc_analytical = 1.0 / c**2

print(f"df/da: 数值={df_da:.6f}, 解析={df_da_analytical:.6f}")
print(f"df/db: 数值={df_db:.6f}, 解析={df_db_analytical:.6f}")
print(f"df/dc: 数值={df_dc:.6f}, 解析={df_dc_analytical:.6f}")
```

运行后你会发现数值近似与解析结果高度吻合。这种验证方法不仅仅是一个数学练习——在实现自动微分引擎时，当你对某个操作的 `_backward` 实现感到不确定，数值梯度检验就是最终的裁判。


---

## 1.3 构建 Value 类：计算图的节点

micrograd 的全部魔法都浓缩在一个类中：`Value`。这个类是计算图的节点，它既存储标量值，也存储该值对某个标量输出（通常是损失函数）的梯度，还记录了它与其他节点的依赖关系。理解 `Value` 类的每个细节，就等于理解了自动微分的核心机制。

### 1.3.1 Value 类的数据结构

让我们先看 `Value` 类的初始化方法和辅助方法：

```python
class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data           # 标量值本身
        self.grad = 0              # 梯度，初始为0
        # internal variables used for autograd graph construction
        self._backward = lambda: None  # 反向传播函数，默认不做任何事
        self._prev = set(_children)    # 产生这个值的父节点集合
        self._op = _op                 # 产生这个值的操作符，用于可视化/debug

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

`Value` 对象有五个核心属性。`data` 存储标量数值本身，比如 `3.0` 或 `-1.5`。`grad` 存储梯度——这个 `Value` 对最终损失函数的敏感度，初始为0，在反向传播后被填充。`_prev` 是一个集合，包含所有参与计算这个值的父 `Value` 节点，用来构建计算图的边。`_op` 是一个字符串，记录是什么操作产生了这个值（如 `'+'`、`'*'`、`'tanh'`），对调试和可视化计算图很有用。

最关键的字段是 `_backward`，它是一个函数（初始为空操作 `lambda: None`）。这个函数的语义是：**当我知道了自己的梯度 `self.grad` 之后，如何将梯度传递给父节点**。每个运算在创建输出节点时，都会为这个输出节点编写一个专属的 `_backward` 函数——这个函数就是一个闭包（closure），它"记住"了父节点是谁以及应该用什么局部导数来传播梯度。

### 1.3.2 实现四则运算

现在我们来逐个实现基本运算。首先是加法：

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
```

这段代码虽然不长，但蕴含了自动微分的核心设计模式。当我们写 `a + b` 时，Python 会调用 `a.__add__(b)`。这个方法做了三件事：第一，将 `other` 包装为 `Value`（如果它还不是的话），这样我们就可以对 `Value` 和原始数字进行混合运算；第二，创建一个新的 `Value` 节点 `out`，其 `data` 是两个操作数之和，`_prev` 记录了 `self` 和 `other` 是它的父节点，`_op` 标记为 `'+'`；第三，定义 `out` 的 `_backward` 函数。

`_backward` 的实现基于一个简单但关键的观察：如果 `out = self + other`，那么 `out` 对 `self` 的局部导数是 1——`self` 增加一点点，`out` 增加完全相同的量。同理，`out` 对 `other` 的局部导数也是 1。根据链式法则，为了将梯度传递给父节点，我们需要将 `out.grad`（`out` 对最终损失的影响）乘以局部导数（都是1），然后累加到父节点的梯度上。这就是为什么 `self.grad += out.grad` 和 `other.grad += out.grad` 都直接使用 `out.grad` 而无需任何缩放。

这里必须注意 `+=`（累加）而非 `=`（赋值）的重要性。假设一个值 `a` 被用于多个计算，比如 `b = a + a`。此时 `a` 通过两条路径影响 `b`，其总梯度应该是两条路径贡献之和。如果用 `=`，`b._backward` 的两次调用中后一次会覆盖前一次的结果；用 `+=` 才能正确累加。这个细节是很多自动微分初学者最容易踩的坑。

接下来是乘法：

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
```

乘法的反向传播逻辑稍有不同。如果 `out = self * other`，根据乘法求导法则，`d(out)/d(self) = other.data`，`d(out)/d(other) = self.data`。因此，传递给 `self` 的梯度是 `other.data * out.grad`，传递给 `other` 的梯度是 `self.data * out.grad`。你可以这样记忆：乘法节点的梯度传播是"对侧值缩放"——`self` 获得 `other` 的值作为缩放因子，`other` 获得 `self` 的值作为缩放因子。

接下来是幂运算，这是实现 `ReLU` 等激活函数的基础：

```python
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
```

这里我们只支持整数或浮点数次幂（不支持 `Value` 作为指数，因为那需要实现 `exp` 和 `log`，我们留作课后练习）。根据幂函数求导法则，$d/dx[x^n] = n \cdot x^{n-1}$，所以局部导数是 `other * self.data ** (other - 1)`，再乘以 `out.grad` 完成链式法则传播。

基于这些基本运算，我们可以通过组合来支持更多运算，而不需要为每个运算都从头写 `_backward`：

```python
    def __neg__(self):       # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1
```

`__neg__` 利用乘法实现取反；`__sub__` 利用加法和取反实现减法；`__truediv__` 利用乘法和幂运算实现除法（因为 `a / b = a * b^{-1}`）。这种组合式设计的优雅之处在于：我们只需要正确实现少数几个基本运算的 `_backward`，更复杂的运算自然继承正确的梯度传播行为。

### 1.3.3 实现激活函数：tanh

神经网络的核心特征之一是**非线性激活函数**。如果没有非线性，多层神经网络的表达能力与单层线性模型完全等价——无论堆叠多少层，最终结果都只是输入的线性组合。非线性激活函数打破了这种限制，使网络能够学习任意复杂的函数映射。

micrograd 支持 `tanh` 作为激活函数（课程中手动推导的部分使用 `tanh`，而 `nn.py` 中使用更简单的 `ReLU`）。`tanh`（双曲正切函数）的数学定义是：

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

`tanh` 的输出范围在 -1 到 1 之间，呈 S 形曲线。它有一个非常优美的导数性质：

$$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$

这意味着，只要你知道 `tanh(x)` 的输出值，就可以直接计算出导数，而无需重新计算指数函数。在代码中，这通常写作 `1 - out.data ** 2`，其中 `out.data` 就是 `tanh(x)` 的输出值。

虽然在 engine.py 的完整实现中我们主要依赖 `ReLU`（它的实现更简单、计算更高效），但课程中手动推导反向传播的核心例子正是基于 `tanh`。理解 `tanh` 的导数公式，对于掌握激活函数在反向传播中的角色至关重要。在后面的手动验证一节中，我们会用完整的数值例子展示 `tanh` 的梯度是如何一层层传播回权重参数的。

现在，我们来补充 `ReLU` 的实现——它是现代神经网络中最常用的激活函数，形式极其简单：`ReLU(x) = max(0, x)`：

```python
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
```

`ReLU` 的反向传播逻辑直观而高效：如果输入为正，梯度原样传递（局部导数为1）；如果输入为负或零，梯度被截断（局部导数为0）。`(out.data > 0)` 在 Python 中返回布尔值，在数值运算中自动转换为 1 或 0，因此这行代码优雅地实现了"正通负断"的行为。`ReLU` 的简洁性使它在大型神经网络中广受欢迎——前向传播只需要一个比较操作，反向传播只需要一个条件判断。


---

## 1.4 反向传播：计算梯度的核心算法

我们已经构建了计算图的节点（`Value` 类）和基本运算，但这些运算各自只定义了局部导数规则——它们知道如何将梯度从输出传播到直接输入。真正的挑战在于：**如何将这些局部规则串联起来，自动计算任意复杂表达式中每个参数的梯度？** 这就是反向传播算法要解决的问题。

### 1.4.1 拓扑排序：确定计算顺序

反向传播从输出节点开始，将梯度沿着计算图的边反向传播回所有输入节点。但这里有一个微妙的问题：当我们为一个节点计算梯度时，必须确保**该节点所有后继节点的梯度已经计算完毕**。换句话说，梯度必须先"流出"子节点，才能"流入"父节点。

这恰好对应了图论中的一个经典问题——**拓扑排序（topological sort）**。拓扑排序将 DAG 的节点排成一个线性序列，使得对于每条有向边 $u \to v$，节点 $u$ 在序列中总是出现在节点 $v$ 之后。应用到我们的场景：如果 `out` 依赖于 `a` 和 `b`（即 `a → out` 和 `b → out`），那么拓扑排序会确保 `out` 排在 `a` 和 `b` 之前。这样，当我们从拓扑序列的末尾倒序遍历时，就保证了每个节点的梯度在其父节点需要它之前已经计算完成。

micrograd 中使用深度优先搜索（DFS）来实现拓扑排序：

```python
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
```

这段代码的递归逻辑是：对于一个节点 `v`，先递归地将它的所有子节点（`v._prev`）加入拓扑序列，然后再将 `v` 自己加入序列。这样，`v` 一定排在它的所有子节点之后。递归的终止条件是节点已经被访问过（`v in visited`），这避免了重复处理和图中的环路（虽然我们假设图是无环的）。

拓扑排序的结果 `topo` 是一个列表，满足这样的性质：如果从左到右遍历，每个节点都出现在它的所有后继节点之前；如果从右到左遍历（即 `reversed(topo)`），每个节点都出现在它的所有前驱节点之后——这正是反向传播需要的顺序。

### 1.4.2 backward() 的实现：从输出到输入传播梯度

有了拓扑排序，反向传播的完整实现就可以写出来了：

```python
    def backward(self):
        # ... (build_topo 代码如上) ...

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

这段代码只有三行核心逻辑，但每一行都有深意。

首先，`self.grad = 1` 设置输出节点（通常代表损失函数值）的梯度为 1。为什么是 1？因为我们在计算 "损失函数对损失函数的梯度"——$dL/dL = 1$。这个初始梯度会像一颗种子，沿着链式法则的链条一路传播，经过每个节点时被局部导数缩放，最终在输入节点处汇聚为 $dL/dx_i$。

然后，`for v in reversed(topo): v._backward()` 按照拓扑排序的逆序遍历每个节点，调用其 `_backward` 函数。由于拓扑排序保证了每个节点的后继节点一定排在它前面，逆序遍历时每个节点的后继一定已经被处理过了——这意味着 `v.grad` 已经被正确计算，可以安全地传递给 `v` 的父节点。

这一简洁的循环完成了一个惊人的壮举：无论计算图有多么复杂，无论参数之间有多少层复合关系，它都能自动计算出损失函数对每个参数的梯度。这就是反向传播的力量所在。

### 1.4.3 梯度累积：同一个节点的多条路径贡献

在继续之前，我们必须强调一个极易被忽视但又极其关键的细节：**梯度需要累加（`+=`），而不是赋值（`=`）**。

考虑表达式 `b = a + a`。这里 `a` 被使用了两次，通过两条完全相同的边连接到输出 `b`。根据链式法则的多路径规则，`a` 对 `b` 的总影响是两条路径贡献的和。在 `_backward` 中，如果写成 `self.grad = out.grad`，第二次调用 `_backward` 时会覆盖第一次的贡献，导致梯度丢失一半；只有写成 `self.grad += out.grad` 才能正确累加。

这个 bug 在真实神经网络中非常常见。想象一个权重被多个输入共享，或者一个偏置被多个神经元共享——如果不累加梯度，这些共享参数的更新就会出错，导致训练过程不稳定或根本不收敛。Karpathy 在课程中特别强调了这一点，称这是他见过的最常见的自动微分 bug。

### 1.4.4 手动验证：数值梯度检验

理论讲得再多，最终也要通过实践来验证。让我们用一个具体的例子来手动追踪反向传播的每一步，并通过数值方法验证其正确性。

假设我们有以下计算图，它模拟了一个简单的神经网络前向传播：

$$n = x_1 \cdot w_1 + x_2 \cdot w_2 + b$$
$$o = \tanh(n)$$

其中 $x_1 = 2.0, w_1 = -3.0, x_2 = 0.0, w_2 = 1.0, b = 6.881373587$（`b` 的特殊取值使得 `tanh` 的输出恰好接近 0.5，便于验证）。

```python
from micrograd.engine import Value

# inputs
x1 = Value(2.0)
x2 = Value(0.0)
# weights
w1 = Value(-3.0)
w2 = Value(1.0)
# bias
b = Value(6.8813735870195432)

# forward pass
x1w1 = x1 * w1          # 2.0 * -3.0 = -6.0
x2w2 = x2 * w2          # 0.0 * 1.0 = 0.0
x1w1x2w2 = x1w1 + x2w2  # -6.0 + 0.0 = -6.0
n = x1w1x2w2 + b        # -6.0 + 6.881... = 0.881...
o = n.tanh()            # tanh(0.881...) ≈ 0.7071

print(f"o = {o.data:.4f}")  # 应输出约 0.7071
```

现在让我们手动执行反向传播。首先 `o.grad = 1.0`（基础情形）。然后：

```python
# 手动反向传播
o.grad = 1.0
# do/dn = 1 - tanh(n)^2 = 1 - o.data^2 ≈ 1 - 0.5 = 0.5
n.grad = (1 - o.data ** 2) * o.grad

# n = x1w1x2w2 + b, 加法节点的局部导数都是1
x1w1x2w2.grad = n.grad * 1  # 1 * 0.5 = 0.5
b.grad = n.grad * 1         # 1 * 0.5 = 0.5

# x1w1x2w2 = x1w1 + x2w2, 又是加法
x1w1.grad = x1w1x2w2.grad * 1  # 0.5
x2w2.grad = x1w1x2w2.grad * 1  # 0.5

# x1w1 = x1 * w1, 乘法: x1的梯度 = w1.data * x1w1.grad
x1.grad = w1.data * x1w1.grad  # -3.0 * 0.5 = -1.5
w1.grad = x1.data * x1w1.grad  # 2.0 * 0.5 = 1.0

# x2w2 = x2 * w2
x2.grad = w2.data * x2w2.grad  # 1.0 * 0.5 = 0.5
w2.grad = x2.data * x2w2.grad  # 0.0 * 0.5 = 0.0
```

这些手动计算的结果是：$w_1$ 的梯度约为 1.0，$w_2$ 的梯度约为 0.0，$b$ 的梯度约为 0.5。它们的含义是：增加 $w_1$ 会使 `o` 增加（梯度为正），增加 $w_2$ 对 `o` 没有影响（因为 $x_2 = 0$），增加 $b$ 会使 `o` 增加。

现在用数值方法来验证：给每个参数一个微小的扰动 $h = 0.001$，观察 `o` 的变化。

```python
# 数值梯度验证
h = 0.001

def forward(x1_v, x2_v, w1_v, w2_v, b_v):
    """封装前向传播为纯函数"""
    xx1, xx2 = Value(x1_v), Value(x2_v)
    ww1, ww2 = Value(w1_v), Value(w2_v)
    bb = Value(b_v)
    oo = (xx1 * ww1 + xx2 * ww2 + bb).tanh()
    return oo.data

x1_v, x2_v, w1_v, w2_v, b_v = 2.0, 0.0, -3.0, 1.0, 6.881373587

# w1 的数值梯度
grad_w1_numerical = (forward(x1_v, x2_v, w1_v + h, w2_v, b_v) -
                     forward(x1_v, x2_v, w1_v - h, w2_v, b_v)) / (2 * h)
print(f"w1.grad: 解析={w1.grad:.4f}, 数值={grad_w1_numerical:.4f}")
# 输出: w1.grad: 解析=1.0000, 数值≈1.0000

# b 的数值梯度
grad_b_numerical = (forward(x1_v, x2_v, w1_v, w2_v, b_v + h) -
                    forward(x1_v, x2_v, w1_v, w2_v, b_v - h)) / (2 * h)
print(f"b.grad:  解析={b.grad:.4f}, 数值={grad_b_numerical:.4f}")
# 输出: b.grad:  解析=0.5000, 数值≈0.5000
```

解析梯度与数值梯度高度吻合，验证了我们的 `_backward` 实现是正确的。这种"手动追踪 + 数值验证"的双重检查方法，是确保反向传播实现无误的黄金标准。在实际开发中，每当你实现了一个新的运算或怀疑某个地方出错了，都可以用这种数值梯度检验来排查问题。


---

## 1.5 从计算图到神经网络

micrograd 的 `engine.py` 提供了构建任意计算图的能力，但直接用 `Value` 来搭建神经网络会非常繁琐。我们需要更高层次的抽象：神经元（Neuron）、层（Layer）、多层感知机（MLP）。这些抽象层封装了神经网络的结构细节，让我们可以用声明式的方式构建网络，而不必手动管理每一个 `Value` 对象。

### 1.5.1 单个神经元的数学表达

**神经元（Neuron）**是神经网络的基本计算单元。它接收多个输入 $x_1, x_2, \ldots, x_n$，每个输入乘以一个权重 $w_i$，然后求和并加上一个偏置 $b$，最后通过一个非线性**激活函数（activation function）**得到输出：

$$\text{out} = \text{activation}\left(\sum_{i=1}^{n} w_i \cdot x_i + b\right)$$

为什么需要非线性激活？这是一个关键问题。假设我们没有激活函数，或者使用线性激活（即 `activation(x) = x`），那么一个神经元的输出就是 $\sum w_i x_i + b$，这本身就是输入的线性组合。如果我们将多个这样的层堆叠起来，每一层都是线性变换，那么整个网络仍然只是输入的线性函数——无论堆叠多少层，其表达能力与单层网络完全相同。这是因为**线性函数的复合仍然是线性函数**。

非线性激活函数的引入打破了这种限制。它让每个神经元能够在输入空间中引入"弯曲"的决策边界，多层这样的弯曲边界组合起来，网络就能够逼近任意复杂的非线性函数。这是神经网络强大表达能力的根源——也是为什么它被称为**通用函数逼近器（universal function approximator）**的理论基础。

micrograd 中使用 `ReLU` 作为默认激活函数。`ReLU(x) = max(0, x)` 的图像像是一个折线：当输入为负时输出0，当输入为正时输出等于输入本身。`ReLU` 有两个显著优点：计算极其简单（前向只需比较，反向只需判断），且在正半区梯度恒为1，不会出现梯度消失问题。当然，它也有缺点——当输入持续为负时，神经元会"死亡"（梯度始终为0，不再学习），后续课程中我们会讨论如何用 `LeakyReLU` 或 batch normalization 来缓解这个问题。

### 1.5.2 Neuron 类的实现

让我们来看单个神经元的代码实现：

```python
import random
from micrograd.engine import Value

class Module:
    """神经网络模块的基类，所有层和模型都继承自它"""

    def zero_grad(self):
        """将所有参数的梯度清零"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """返回该模块包含的所有参数，子类需要重写"""
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        # nin: 输入特征的数量
        # w: 随机初始化的权重，范围在 [-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin  # 是否使用非线性激活

    def __call__(self, x):
        # 计算加权和: sum(wi * xi) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # 如果 nonlin=True, 通过 ReLU 激活; 否则直接输出
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
```

`Neuron` 类有几个值得注意的设计选择。首先是权重初始化：`random.uniform(-1, 1)` 将权重随机初始化在 -1 到 1 之间。这种简单的均匀分布在小型网络中工作良好，但对于更深的网络，更精细的初始化方法（如 Xavier 初始化或 He 初始化）能够显著改善训练的稳定性——这些我们会在后续课程中详细介绍。偏置 `b` 初始化为0是常见的做法，因为对称性破缺（symmetry breaking）主要由随机权重提供。

`__call__` 方法实现了前向传播。`sum((wi * xi for wi, xi in zip(self.w, x)), self.b)` 这行代码利用 Python 的 `sum` 函数的可选起始参数，将偏置 `self.b` 作为求和的初始值，这样 `b` 自然地参与了整个求和过程，无需在求和完成后再单独添加。`act.relu()` 在 `act.data < 0` 时输出 0，否则输出 `act.data` 本身。

`parameters()` 方法返回所有需要训练的参数（权重列表加偏置），这个设计使得我们可以方便地获取网络中所有可学习的参数，用于梯度清零和参数更新。

### 1.5.3 Layer 类：一组并行神经元

单个神经元只能产生一个标量输出。为了提取多个特征，我们需要将多个神经元**并行**排列，组成一个**层（Layer）**。

```python
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        # nin: 输入维度, nout: 输出维度（即神经元的数量）
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]  # 每个神经元独立计算
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
```

`Layer` 的核心是一个神经元列表。当数据传入层时，每个神经元独立地处理同样的输入 `x`，但产生不同的输出——因为每个神经元有自己独立的权重和偏置。如果层只有一个输出神经元（`nout=1`），`__call__` 会直接返回单个标量而不是长度为1的列表，这是一个方便使用的小优化。

### 1.5.4 MLP 类：堆叠多层构建深度网络

**多层感知机（MLP, Multi-Layer Perceptron）**是将多个层**串行**堆叠而成的神经网络。数据从输入层流入，经过一层又一层的变换，最终从输出层产生预测结果。

```python
class MLP(Module):

    def __init__(self, nin, nouts):
        # nin: 输入维度
        # nouts: 列表，定义每层的输出维度
        # 例如 nouts=[4, 4, 1] 表示：
        #   第1层: nin -> 4,  第2层: 4 -> 4,  第3层: 4 -> 1
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # 逐层传递
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
```

`MLP` 的构造函数有一个关键细节：`nonlin=(i != len(nouts)-1)` 确保**最后一层不使用非线性激活**。为什么？因为在二分类或回归任务中，最后一层通常需要输出一个无约束的实数值（logit），而非经过 `ReLU` 截断后的非负数。如果最后一层也应用 `ReLU`，输出就永远不可能为负，这会严重限制模型的表达能力。这个设计决策在实际深度学习框架中是标准做法：隐藏层使用激活函数引入非线性，输出层根据任务类型选择适当的输出方式（对于二分类可能用 sigmoid，多分类用 softmax，回归则不用激活）。

让我们看一个具体的网络创建示例：

```python
from micrograd.nn import MLP

# 创建一个 MLP: 3个输入 -> 4神经元隐藏层 -> 4神经元隐藏层 -> 1个输出
n = MLP(3, [4, 4, 1])
print(n)
# 输出: MLP of [Layer of [ReLUNeuron(3), ... x4], Layer of [ReLUNeuron(4), ... x4], Layer of [LinearNeuron(4)]]

# 查看参数数量
print(f"参数总数: {len(n.parameters())}")
# 输出: 参数总数: 41
# 计算: (3*4 + 4) + (4*4 + 4) + (4*1 + 1) = 16 + 20 + 5 = 41
```

这个网络有41个可训练参数。第一层有4个神经元，每个有3个权重和1个偏置，共 $4 \times (3 + 1) = 16$ 个参数；第二层有4个神经元，每个有4个权重和1个偏置，共 $4 \times (4 + 1) = 20$ 个参数；第三层有1个神经元，4个权重和1个偏置，共5个参数。总计 $16 + 20 + 5 = 41$。

---

## 1.6 训练循环

有了自动微分引擎和神经网络模块，我们终于来到了最令人兴奋的部分——训练。训练循环是神经网络学习的核心机制，它由四个步骤组成，反复迭代直至收敛：前向传播计算预测值，损失函数评估预测与真实的差距，反向传播计算梯度，参数更新沿梯度反方向微调权重。

### 1.6.1 损失函数：MSE 衡量预测差距

**损失函数（loss function）**量化模型预测与真实目标之间的差距。在 micrograd 的示例中，我们使用 **均方误差（MSE, Mean Squared Error）**：

$$L = \sum_{i} (y_{\text{pred},i} - y_{\text{true},i})^2$$

MSE 对每个样本的预测误差求平方后累加。平方的作用有两个：一是确保误差总是正的（避免正负误差相互抵消），二是对大误差给予更大的惩罚（因为平方会放大偏差）。在二分类任务中，当目标值是 +1 或 -1 时，MSE 驱使网络输出尽可能接近这些目标值。

### 1.6.2 前向传播

前向传播将输入数据通过网络得到预测值：

```python
# 数据集: 4个样本，每个样本3个特征
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# 标签: 对应的目标输出
ys = [1.0, -1.0, -1.0, 1.0]

# 初始化网络
n = MLP(3, [4, 4, 1])

# 训练循环
for k in range(100):
    # 前向传播: 对每条数据计算预测值
    ypred = [n(x) for x in xs]
    # ypred 是4个 Value 对象的列表
```

`ypred = [n(x) for x in xs]` 这行代码对数据集中的每个样本执行一次前向传播。每次调用 `n(x)`，数据从输入层流入，经过两个 ReLU 隐藏层，最终从输出层产生一个标量预测值。这个过程构建了一张完整的计算图——每个 `Value` 对象都记录了它与其他 `Value` 的依赖关系，为后续的反向传播做好了准备。

### 1.6.3 反向传播

```python
    # 计算损失: 均方误差
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # 反向传播
    n.zero_grad()
    loss.backward()
```

损失 `loss` 是一个 `Value` 对象，它聚合了所有预测误差的平方。当我们调用 `loss.backward()` 时，micrograd 首先对计算图进行拓扑排序，然后从 `loss` 节点开始反向传播梯度。最终，网络中的每个参数的 `grad` 字段都被填充了损失函数对该参数的偏导数。

这里有一个极易出错的细节：在调用 `loss.backward()` 之前，必须先执行 `n.zero_grad()` 将上一轮迭代留下的梯度清零。为什么要清零？因为 `Value.grad` 是**累加**的——每一轮 `backward()` 都会将新计算的梯度加到 `grad` 上。如果不先清零，第二轮的梯度会叠加在第一轮的梯度之上，第三轮叠加在前两轮之上，如此往复。这会导致梯度爆炸（迅速变得极大），模型根本无法训练。

这个 bug 的本质是：反向传播算法假设每个参数的 `grad` 从0开始累加，因为同一个参数可能通过多条路径接收梯度贡献。如果我们不手动清零，`grad` 就永远只增不减，失去了意义。

### 1.6.4 参数更新：梯度下降

```python
    # 参数更新: 沿梯度反方向微调
    for p in n.parameters():
        p.data -= 0.1 * p.grad

    print(f"step {k}, loss {loss.data}")
```

梯度下降的规则极其简单：`p.data -= learning_rate * p.grad`。其中 `learning_rate = 0.1` 是**学习率（learning rate）**，控制每次更新的步长。学习率是一个关键的超参数：太大可能导致在损失曲面上"跳跃"而无法收敛，甚至发散；太小则收敛极其缓慢。0.1 这个值对于这个微型示例是合理的，但对于更复杂的网络和数据集，通常需要使用更小的学习率（如 0.01、0.001），并配合学习率衰减策略。

这行代码的直觉直接来自导数的定义：如果 `p.grad` 为正，说明增加 `p` 会使损失增加，因此我们需要减小 `p`（减去一个正值）；如果 `p.grad` 为负，说明增加 `p` 会使损失减小，减去一个负值等于增大 `p`。无论哪种情况，更新方向都是使损失下降的方向。

将以上步骤组合起来，完整的训练循环如下：

```python
from micrograd.nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

for k in range(100):
    # 前向传播
    ypred = [n(x) for x in xs]

    # 计算损失
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # 反向传播
    n.zero_grad()
    loss.backward()

    # 参数更新
    for p in n.parameters():
        p.data -= 0.1 * p.grad

    print(f"step {k}, loss {loss.data:.6f}")

# 训练结束后检查预测
print("\n最终预测:")
for x, ygt in zip(xs, ys):
    ypred = n(x)
    print(f"  输入 {x} -> 预测 {ypred.data:+.4f}, 目标 {ygt:+.1f}")
```

运行这段代码，你会看到损失值从初始的较高值（可能大于1）逐步下降，经过约20-50步迭代后收敛到接近0。最终，四个样本的预测值会非常接近它们的目标值（+1.0 或 -1.0）。这意味着网络成功地"记住"了这四条数据——当然，这只是一个微型演示，真实的深度学习任务涉及数百万条数据和更复杂的模型。

### 1.6.5 常见 bug：为什么 zero_grad 如此重要

让我们用一个简化的例子来直观感受梯度未清零的后果：

```python
# 演示: 如果忘记 zero_grad 会发生什么
n = MLP(3, [4, 4, 1])

for k in range(5):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # 错误: 没有调用 n.zero_grad()
    loss.backward()

    for p in n.parameters():
        print(f"  step {k}, param grad magnitude: {abs(p.grad):.4f}")
```

你会观察到梯度的量级在每一轮都在增长，因为每一轮的新梯度都被累加到之前轮的梯度上。经过几轮迭代后，梯度变得极大，参数更新步长也异常巨大，导致损失值可能不降反升，模型完全失控。

在 PyTorch 中，这个模式是完全一致的：每次 `loss.backward()` 之前都需要调用 `optimizer.zero_grad()`（或手动将 `param.grad` 设为 `None`）。这个看似简单的操作是训练稳定性的基本保障，也是每个深度学习从业者必须牢记的习惯。


---

## 1.7 课后练习与学习资源

### 1.7.1 课后练习题

以下练习分为三个层次。**基础练习**帮助你巩固核心概念，**进阶练习**拓展 micrograd 的功能，**挑战练习**连接工业级框架。

#### 基础练习：导数与数值验证

**练习 1：解析梯度推导**

给定函数 $f(a, b, c) = -a^3 + \sin(3b) - 1.0/c + b^{2.5} - a^{0.5}$，手动推导它对 $a, b, c$ 的解析偏导数，然后在 `a=2.0, b=3.0, c=4.0` 处用 Python 计算其数值。

提示：逐项求导。$a$ 出现在两项中（$-a^3$ 和 $-a^{0.5}$），根据偏导数的加法规则，对 $a$ 的偏导数是这两项导数之和。

**练习 2：前向差分数值梯度**

不使用任何微积分公式，仅用数值方法估计 $f$ 对 $a$ 的偏导数：

$$\frac{\partial f}{\partial a} \approx \frac{f(a+h, b, c) - f(a, b, c)}{h}$$

其中 $h = 10^{-6}$。将结果与练习 1 的解析结果对比，计算相对误差。

**练习 3：对称导数**

使用**对称导数**（Symmetric Derivative）获得更精确的数值近似：

$$\frac{\partial f}{\partial a} \approx \frac{f(a+h, b, c) - f(a-h, b, c)}{2h}$$

比较对称导数与前向差分的精度差异。你会发现对称导数的误差通常比前向差分小两个数量级——这是因为对称导数的截断误差是 $O(h^2)$，而前向差分是 $O(h)$。

#### 进阶练习：扩展 micrograd

**练习 4：实现 `exp()` 和 `log()`**

扩展 `Value` 类以支持指数运算和自然对数运算，并实现正确的反向传播。

提示：`exp(x)` 的导数是 `exp(x)` 本身；`log(x)` 的导数是 `1/x`。你的实现应该遵循与 `__add__`、`__mul__` 相同的设计模式：创建输出节点，定义 `_backward` 闭包，利用链式法则传播梯度。

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
        self.grad += out.data * out.grad  # d/dx exp(x) = exp(x)
    out._backward = _backward
    return out

def log(self):
    out = Value(math.log(self.data), (self,), 'log')
    def _backward():
        self.grad += (1.0 / self.data) * out.grad  # d/dx log(x) = 1/x
    out._backward = _backward
    return out
```

**练习 5：Softmax 与负对数似然损失**

基于练习 4 实现的 `exp()` 和 `log()`，实现 `softmax` 函数和负对数似然损失（negative log likelihood loss），并验证梯度计算的正确性。

```python
def softmax(logits):
    counts = [logit.exp() for logit in logits]
    denominator = sum(counts)
    return [c / denominator for c in counts]

logits = [Value(0.0), Value(3.0), Value(-2.0), Value(1.0)]
probs = softmax(logits)
loss = -probs[3].log()  # 假设正确类别是第 3 类

loss.backward()

# 打印每个 logit 的梯度
for i, logit in enumerate(logits):
    print(f"logit[{i}].grad = {logit.grad:.6f}")
```

Softmax 的梯度有一个简洁的解析形式：对于正确类别 $j$，$\partial L / \partial z_i = p_i - 1_{i=j}$（其中 $1_{i=j}$ 在 $i=j$ 时为1，否则为0）。验证你的自动微分结果与这个公式一致。

**练习 6：与 PyTorch 对比**

用 PyTorch 重写练习 5 的代码，创建对应的 `torch.Tensor` 对象，设置 `requires_grad=True`，计算相同的损失和梯度。比较 PyTorch 的梯度值与你自己的 micrograd 实现是否一致。如果不一致，排查是哪一个运算的 `_backward` 实现有误——数值梯度检验是你最好的朋友。

#### 挑战练习

**练习 7：支持高阶导数**

micrograd 的 `Value` 类天然支持高阶导数计算——因为梯度本身也是 `Value` 对象，你可以对梯度再求梯度。尝试计算一个简单函数（如 $f(x) = x^3$）的二阶导数和三阶导数，验证结果与解析解一致。

**练习 8：实现 L2 正则化**

在训练循环中加入 L2 正则化项：$L_{\text{total}} = L_{\text{MSE}} + \lambda \sum_{p} p^2$。L2 正则化通过在损失中加入权重的平方和，鼓励权重保持较小，从而防止过拟合。实验不同的正则化强度 $\lambda$（如 0.001、0.01、0.1），观察对训练过程和最终模型行为的影响。

---

### 1.7.2 推荐学习资源

#### 核心资源

- [YouTube 视频 — building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Karpathy 亲自讲授的课程视频，时长约 2.5 小时，强烈推荐至少完整观看一次。视频中的讲解节奏、可视化展示和即兴调试都是教科书无法替代的学习体验 [^1^]。
- [micrograd GitHub 仓库](https://github.com/karpathy/micrograd) — 完整的自动微分引擎代码，包含 `engine.py`（核心自动微分）和 `nn.py`（神经网络模块）。仓库的 README 中有一段极简的示例代码，展示了整个训练流程 [^2^]。
- [Jupyter Notebook — 上半部分](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb) — 涵盖导数、链式法则、手动反向传播和 `Value` 类的构建过程 [^3^]。
- [Jupyter Notebook — 下半部分](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb) — 涵盖 `Module` 基类、`Neuron`、`Layer`、`MLP` 的实现和完整训练循环 [^4^]。

#### 练习题与解答

- [micrograd 练习题 Notebook](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/Exercises/micrograd%20exercises/micrograd_exercises.ipynb) — 社区整理的完整练习题，覆盖从导数计算到 softmax 实现的多个难度层次 [^5^]。
- [社区笔记与解答 — MK2112/nn-zero-to-hero-notes](https://github.com/MK2112/nn-zero-to-hero-notes) — 详尽的课程笔记和练习题解答，适合在遇到困难时参考 [^6^]。

#### 深度阅读

- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Karpathy 的经典文章，论述为什么即使在使用 PyTorch 这样的高级框架时，深入理解反向传播仍然至关重要。文中列举了多个因为不理解反向传播而导致的常见 bug，与我们在课程中讨论的 `zero_grad` 和梯度累积问题遥相呼应 [^7^]。
- [反向传播历史论文 — Rumelhart et al. (1986)](https://www.nature.com/articles/323533a0) — 反向传播算法的原始论文，发表于 *Nature* 杂志。虽然年代久远，但它是深度学习历史上的里程碑，值得所有认真学习的从业者至少浏览一次 [^8^]。
- [Wikipedia — Chain Rule](https://en.wikipedia.org/wiki/Chain_rule) — 链式法则的数学定义和多种证明方法 [^9^]。
- [Wikipedia — Symmetric Derivative](https://en.wikipedia.org/wiki/Symmetric_derivative) — 对称导数的数学性质，比前向差分更精确的数值微分方法 [^10^]。
- [WolframAlpha](https://www.wolframalpha.com/) — 在线数学计算工具，可用于验证复杂函数的导数解析表达式 [^11^]。

#### 其他语言的 micrograd 实现

阅读不同编程语言实现的 micrograd 是深化理解的有效方法。当你需要用另一种语言重写同一个算法时，每个细节都会被重新审视：

- [ferric-micrograd](https://github.com/msminhas93/ferric-micrograd) — Rust 实现，展示了所有权和借用系统在自动微分中的应用 [^12^]。
- [micrograd-cpp-2023](https://github.com/kfish/micrograd-cpp-2023) — C++ 实现，展示了手动内存管理下的计算图构建 [^13^]。

#### 后续课程预告

完成本课的学习后，你已经掌握了深度学习的核心机制——自动微分和反向传播。下一课我们将进入语言建模的世界，从一个简单的 **bigram 字符级语言模型**开始，学习如何用计数方法和神经网络来预测名字中下一个字符的概率分布。在那里，你将第一次接触 PyTorch 的 `Tensor`，理解向量化计算的重要性，并探索**负对数似然损失**和 **softmax** 函数在分类问题中的角色。




---

## 2. 第2课：makemore Part 1 — 语言模型基础与Bigram模型

前一章我们用手工打造的 `micrograd` 引擎深入理解了反向传播的数学本质。现在，让我们将这份理解带入一个崭新的领域：**语言建模（language modeling）**。我们将构建一个名为 `makemore` 的字符级语言模型——它会学习一个名字的统计规律，然后生成更多类似的名字。从本章开始，我们不再手动实现反向传播，而是使用 PyTorch 提供的 `torch.Tensor` 和自动微分（autograd）系统，将注意力集中在模型的设计与训练逻辑上。

这一章的核心是一个看似简单的 **bigram（二元组）模型**：给定一个字符，预测下一个字符是什么。我们会用两种截然不同的方式实现它——先是用统计计数法直接"数出来"，然后用神经网络"学出来"，最终发现两者在数学上的深刻联系。这个联系将为我们后续构建多层感知机（MLP）、卷积网络和Transformer打下关键直觉。

---

### 2.1 语言建模的核心问题

#### 2.1.1 问题定义：给定前文，预测下一个字符的概率分布

语言建模（language modeling）是当代自然语言处理的核心任务。它的数学表述非常简洁：给定一个由字符组成的序列 $x_1, x_2, \ldots, x_{t-1}$，我们希望学习一个模型来预测下一个字符 $x_t$ 的**条件概率分布**（conditional probability distribution）：

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

如果我们能够准确估计这个分布，就可以计算出任意完整序列的**联合概率**（joint probability）：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

这是**链式法则（chain rule）**在概率论中的直接应用——它将一个复杂的多变量联合概率拆解为一系列条件概率的乘积。每一次预测只关心"给定已经看到的内容，下一个字符最可能是什么"。

这种逐个字符构建序列的方式，引出了一个极其重要的生成范式。

#### 2.1.2 自回归生成：训练好的模型可以通过反复采样生成新文本

一旦我们训练好了模型，就获得了一种强大的能力：**自回归生成（autoregressive generation）**。具体过程如下：

1. 从一个特殊的**开始标记（start token）**，记作 `.`，出发；
2. 将当前序列输入模型，得到下一个字符的概率分布 $P(x_t \mid \text{context})$；
3. 从这个分布中**采样（sample）**一个字符；
4. 将采样到的字符追加到序列末尾，回到步骤2；
5. 当采样到**结束标记（end token）**，同样记作 `.`，时停止。

这个过程的核心在于"自回归"（autoregressive）——模型用**自己生成的输出**作为下一步的输入，逐步"写"出一个完整的序列。这正是GPT等大语言模型生成文本的基本原理。我们从bigram这个最简单的模型开始，已经能够窥见这一范式的全貌。

#### 2.1.3 数据集介绍：32,000个英文名字，字符级别的建模任务

本章使用的数据集来自2018年美国社会保障局（ssa.gov）公布的英文名字列表，共计32,033个名字[^1^]。以下是前几个名字：

```python
words = open('names.txt', 'r').read().splitlines()
print(words[:5])   # ['emma', 'olivia', 'ava', 'isabella', 'sophia']
print(len(words))  # 32033
```

这是一个**字符级别（character-level）**的建模任务。我们的模型不处理单词或子词（subword），而是直接在字母的粒度上工作。每个名字由a-z的26个小写字母组成，最短2个字符，最长15个字符。模型需要学习的是：什么样的字母组合构成了一个"像名字"的字符串。

为了让模型知道序列何时开始、何时结束，我们引入一个特殊标记 `.`（dot），它同时扮演**开始标记（start token）**和**结束标记（end token）**的角色。例如，名字 `"emma"` 会被处理为序列 `['.', 'e', 'm', 'm', 'a', '.']`。这个设计比使用 `<S>` 和 `<E>` 两个不同标记更简洁优雅——从 `.` 出发回到 `.`，构成一个完整的生成循环。

---

### 2.2 Bigram模型：计数方法

#### 2.2.1 Bigram假设：下一个字符只取决于前一个字符

Bigram模型做出了一个极为大胆的简化假设——**马尔可夫假设（Markov assumption）**：下一个字符的概率**仅取决于前一个字符**，而与更早的历史完全无关：

$$P(x_t \mid x_1, \ldots, x_{t-1}) \approx P(x_t \mid x_{t-1})$$

这个假设当然远非完美。"emma"中的第二个 `'m'` 并不仅仅取决于第一个 `'m'`——它实际上还取决于前面的 `'e'`。但正是这种极端的简化，让我们能够用最直观的方式理解语言建模的基本框架。后续的章节将逐步放宽这个假设，引入更长的上下文。

#### 2.2.2 统计计数法：遍历数据集，统计每对相邻字符出现的次数

既然bigram假设告诉我们只需要关心"前一个字符 → 下一个字符"的转移规律，最直接的方法就是**数数**——遍历整个数据集，统计每一对相邻字符（bigram）出现的次数。

首先，我们需要建立字符与整数索引之间的映射关系：

```python
import torch

chars = sorted(list(set(''.join(words))))  # 26个字母: ['a', 'b', ..., 'z']
stoi = {s: i + 1 for i, s in enumerate(chars)}  # a=1, b=2, ..., z=26
stoi['.'] = 0  # 特殊标记 . 对应索引0
itos = {i: s for s, i in stoi.items()}  # 反向映射
```

这里 `stoi`（string-to-integer）和 `itos`（integer-to-string）是字符和索引之间的双向查找表。将 `.` 设为索引0是一个惯例，它将作为我们的"第27个"特殊符号。

接下来，创建一个 $27 \times 27$ 的计数矩阵 $N$，其中 $N[i, j]$ 表示"字符 $i$ 后面跟着字符 $j$"的次数：

```python
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']  # 在名字前后添加 . 标记
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

注意 `zip(chs, chs[1:])` 这个巧妙的写法——它将列表与其自身向后偏移一位的版本配对，从而遍历所有相邻的bigram。例如对于 `"emma"`，生成的bigram序列是 `('.', 'e'), ('e', 'm'), ('m', 'm'), ('m', 'a'), ('a', '.')`。

统计完成后，我们可以用matplotlib将这个计数矩阵可视化。图中的每个格子标注了bigram对应的字符组合和出现次数。通过观察这张图，我们可以直接读出数据的统计规律：哪些bigram频繁出现（如 `'n'` 后面跟结束标记，`'a'` 后面跟 `'n'`），哪些几乎从不出现。

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
```

这个可视化本身就蕴含了大量信息。矩阵第一行（索引0，对应 `.`）表示"名字以什么字符开头"的分布；最后一列（索引0，对应 `.`）表示"名字以什么字符结尾"的分布。矩阵中的明亮区域揭示了英文名字中最常见的字符转移模式。

#### 2.2.3 从计数到概率：归一化得到条件概率分布

计数矩阵 $N$ 本身还不是概率分布。为了得到条件概率 $P(x_{\text{next}} \mid x_{\text{current}})$，我们需要对每一行进行**归一化**（normalization），使所有行和为1。

但在此之前，先处理一个关键问题：**模型平滑（model smoothing）**。在训练集中从未出现过的bigram，其计数为0。如果直接用这些0计算概率，任何包含未见过bigram的名字都会被赋予0概率，这在评估损失时会导致灾难性的后果（对数概率趋于负无穷）。一个经典的解决方案是 **Laplace平滑**——给每个计数加上一个小常数（这里取1）：

```python
P = (N + 1).float()  # 加1平滑，避免0概率
P /= P.sum(1, keepdim=True)  # 每行归一化，使行和为1
```

`P.sum(1, keepdim=True)` 对每一行求和，`keepdim=True` 保持维度为 $(27, 1)$ 而非 $(27,)$，这使得后续的除法操作能够通过**广播（broadcasting）**正确执行。如果没有 `keepdim=True`，PyTorch可能会尝试将 $(27, 27)$ 除以 $(27,)$，这会导致一个常见的广播bug——它不会报错，但会按列而非按行归一化，得到完全错误的结果。这个细节我们会在2.4.2节详细讨论。

经过这一步，`P[i, :]` 就是一个有效的概率分布——它非负且和为1，表示"在字符 $i$ 之后，下一个字符是什么"的条件概率。

#### 2.2.4 采样生成：从训练好的bigram分布中采样生成新名字

现在我们可以从模型中**采样（sample）**生成新名字了。采样过程遵循2.1.2节描述的自回归范式：

```python
g = torch.Generator().manual_seed(2147483647)  # 固定随机种子以保证可复现

for i in range(5):
    out = []
    ix = 0  # 从 '.' 开始
    while True:
        p = P[ix]  # 当前字符对应的下一个字符的概率分布
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # 采样到 '.'，结束
            break
    print(''.join(out))
```

这段代码的精髓在于 `torch.multinomial`——它按照给定的概率分布 `p` 进行有放回采样。概率越大的字符被采中的可能性越高，但小概率的字符仍然有机会出现，这为生成结果注入了多样性。固定随机种子让我们能够复现相同的结果，这对于调试和教学至关重要。

典型的生成结果看起来像这样：`junide`、`sahana`、`kohin`……它们明显不是真实的名字，但已经有了名字"应有的样子"——合理的开头、常见的字母组合、适当的长度。这个简单的bigram模型已经捕捉到了英文名字的一些表层统计规律。

我们也可以计算模型在训练集上的**损失（loss）**。对于一个bigram模型，最常用的损失函数是**负对数似然（Negative Log-Likelihood, NLL）**：

```python
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood
print(f'{log_likelihood=}')    # 约为 -559951
print(f'{nll=}')                # 约为 559951
print(f'{nll / n:.4f}')         # 约为 2.4544
```

**平均负对数似然**约为2.45。这个数值有一个直观的概率解释：它等价于说模型给每个正确的bigram分配的平均概率约为 $e^{-2.45} \approx 0.086$。如果我们能把这个损失降到更低（接近0），就意味着模型对每个bigram的预测更加确定。有趣的是，如果我们训练一个完全"无知"的模型——给27个字符均匀分配概率 $1/27 \approx 0.037$，对应的NLL损失就是 $-\log(1/27) \approx 3.296$。我们的bigram模型（2.45）显著优于这个基线，说明它确实学到了数据中的统计规律。

---

### 2.3 Bigram模型：神经网络方法

#### 2.3.1 用神经网络学习bigram：输入前一个字符，输出下一个字符的概率分布

计数法虽然直观有效，但它有一个根本性的局限：每个bigram需要一个独立的计数。当词汇量增大或者上下文变长时，计数矩阵的维度会**指数级爆炸**。更重要的是，计数法无法自然地推广到未见过的组合——平滑只是一个"补丁"，而非根本解决方案。

神经网络提供了一种截然不同的思路。我们不再显式存储每个bigram的计数，而是让模型**学习**从"前一个字符"到"下一个字符的概率分布"之间的映射。这个映射通过一个**可学习的权重矩阵** $W$ 来实现。令人惊讶的是，对于这个特定的bigram任务，神经网络最终学到的结果与计数法殊途同归——但神经网络框架可以自然地扩展到更复杂的模型结构，这是计数法无法做到的。

让我们从数据准备开始：

```python
import torch.nn.functional as F

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])  # 输入：前一个字符的索引
        ys.append(stoi[ch2])  # 目标：下一个字符的索引

xs = torch.tensor(xs)  # 输入张量，形状: (num_bigrams,)
ys = torch.tensor(ys)  # 目标张量，形状: (num_bigrams,)
num = xs.nelement()
print('number of examples:', num)  # 约228,146个bigram
```

`xs` 和 `ys` 分别是我们的输入序列和目标序列。每个训练样本是一个 `(input_char, target_char)` 对：模型看到 `xs[i]`，应该预测 `ys[i]`。

#### 2.3.2 One-hot编码：将字符索引转换为向量表示

神经网络不能直接处理整数索引——它需要数值向量作为输入。对于分类变量，最常用的表示方法是 **one-hot编码**。一个字符的one-hot编码是一个长度为27的向量，其中对应字符索引的位置为1，其余为0。

```python
# xs 中前5个样本的one-hot编码
print(F.one_hot(torch.tensor(0), num_classes=27))
# tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

在训练时，我们对整个输入序列进行one-hot编码：

```python
xenc = F.one_hot(xs, num_classes=27).float()  # 形状: (228146, 27)
```

`xenc`（x-encoded）的形状是 `(228146, 27)`——每一行是一个27维的one-hot向量，对应一个输入字符。为什么要 `.float()` 转换？因为 `F.one_hot` 返回的是整数类型的张量，而后续的矩阵乘法需要浮点数。

#### 2.3.3 可学习的权重矩阵W：27×27的矩阵，每一行代表一个输入字符对应的输出logits

现在来看模型的核心——一个 $27 \times 27$ 的可学习权重矩阵 $W$：

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

`W` 的每一列（共27列）对应一个可能的"下一个字符"；`W` 的每一行（共27行）将通过网络训练，编码一个"输入字符"如何影响所有"下一个字符"的可能性。当输入是字符 $i$ 的one-hot向量时，矩阵乘法 `xenc @ W` 实际上就是从 $W$ 中**选出第 $i$ 行**。这一行包含27个数值，称为 **logits**——它们是softmax之前的原始"分数"，越高表示模型认为对应的下一个字符越可能出现。

为什么叫logits？因为这些值经过指数化和归一化后就变成了概率，而在统计学中，概率的对数比值（log-odds）叫做logit。这里的 $W$ 矩阵可以被理解为存储了"log-counts"（计数的对数）——这正是神经网络与计数法等价性的关键。

前向传播和softmax的完整过程如下：

```python
# 前向传播
xenc = F.one_hot(xs, num_classes=27).float()  # (228146, 27)
logits = xenc @ W  # (228146, 27), 每个样本对应的logits

# Softmax：将logits转换为概率分布
counts = logits.exp()  # (228146, 27), 指数化确保非负
probs = counts / counts.sum(1, keepdims=True)  # (228146, 27), 归一化

# 计算负对数似然损失
loss = -probs[torch.arange(num), ys].log().mean()
```

注意 `probs[torch.arange(num), ys]` 这一行——这是一个巧妙的**高级索引**操作。`torch.arange(num)` 生成 `[0, 1, 2, ..., num-1]`，对应每个训练样本；`ys` 是每个样本的目标字符索引。合在一起，这个索引操作**精确地选出了模型对每个正确下一个字符预测的概率**。然后取对数、取负、求平均，就得到了损失。

这个手动softmax + NLL的实现虽然清晰，但在数值稳定性方面存在隐患。我们在2.4.4节会看到如何用 `F.cross_entropy` 更安全地实现同样的计算。

训练循环遵循我们已经熟悉的前向-反向-更新模式：

```python
for k in range(100):
    # 前向传播
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()

    print(f'step {k}, loss {loss.item():.4f}')

    # 反向传播
    W.grad = None  # 等价于 zero_grad()
    loss.backward()

    # 参数更新
    W.data += -50 * W.grad
```

学习率设为50看似很大，但这是因为我们处理的是大批量的数据（约22.8万个bigram），梯度被大量样本平均后非常小。经过约100轮迭代，损失收敛到约2.45——与计数法的结果几乎完全一致。

#### 2.3.4 等价性证明：当W=log(counts)时，神经网络方法与计数法等价

为什么两个看似截然不同的方法——一个直接数数、一个用梯度下降优化——会得到几乎相同的损失？答案在于一个深刻的数学等价性。

考虑计数法的概率矩阵 $P$：

$$P[i, j] = \frac{N[i, j] + 1}{\sum_k (N[i, k] + 1)}$$

现在，假设神经网络的权重矩阵满足 $W = \log(N + 1)$（即每个元素是平滑后计数的对数）。那么前向传播时，对于输入字符 $i$：

$$\text{logits}[j] = W[i, j] = \log(N[i, j] + 1)$$

$$\text{counts}'[j] = \exp(\text{logits}[j]) = N[i, j] + 1$$

$$\text{probs}'[j] = \frac{N[i, j] + 1}{\sum_k (N[i, k] + 1)}$$

这与计数法的概率 $P[i, j]$ **完全一致**！

这个等价性告诉我们一个重要事实：bigram神经网络本质上是在学习"计数的对数"（log-counts）。梯度下降在寻找最优解的过程中，最终会把 $W$ 推向与 $\log(N + 1)$ 接近的值。当然，由于神经网络还受到初始化、优化过程和可能的正则化影响，它学到的 $W$ 不会与 $\log(N + 1)$ 完全相同——但两者的损失会非常接近。

这个等价性是 `makemore` 系列课程中最优雅的洞察之一。它建立了一种桥梁：计数法是"表查找"，神经网络是"参数化学习"，但在bigram这个简单设定下，两者在数学上是同一回事。随着我们后续引入更长的上下文和更深层的网络，神经网络将展现出计数法无法企及的表达能力——但这种等价性赋予我们信心：**神经网络学习的仍然是数据的统计规律，只是以一种更灵活、更紧凑的方式**。

---

### 2.4 torch.Tensor与梯度优化

#### 2.4.1 torch.Tensor基础：创建、索引、运算，requires_grad=True开启自动微分

在前一章的 `micrograd` 中，我们手动构建了 `Value` 类来追踪梯度和反向传播。现在，PyTorch的 `torch.Tensor` 替我们完成了这一切。

创建一个需要梯度的张量非常简单：

```python
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

`requires_grad=True` 标志告诉PyTorch：对于这个张量的所有运算，请自动构建计算图并准备计算梯度。当调用 `.backward()` 时，PyTorch会自动沿着计算图反向传播，将梯度累加到 `W.grad` 中。

```python
loss.backward()       # 反向传播，计算梯度
print(W.grad)         # 查看梯度
print(W.grad.shape)   # (27, 27)，与W形状相同
```

在每次反向传播之前，需要清零梯度。注意这里使用 `W.grad = None` 而非 `.zero_()`——在PyTorch中这是推荐的做法，更节省内存：

```python
W.grad = None  # 清零梯度
```

**不要**在原地更新 `W`（如 `W -= 50 * W.grad`）——这会破坏计算图，导致后续 `.backward()` 失败。正确的方式是通过 `.data` 属性绕开梯度追踪：

```python
W.data += -50 * W.grad  # 正确的参数更新方式
```

#### 2.4.2 广播机制（broadcasting）：不同形状的Tensor如何进行运算

PyTorch的**广播（broadcasting）**机制允许不同形状的张量之间进行元素级运算，这是向量化代码高效运作的基石。但广播也是bug的高发区，理解它的规则至关重要。

广播的核心规则：

1. 如果两个张量的维度数不同，从后往前比较，维度数为1的维度会被**拉伸**以匹配另一个张量；
2. 如果两个张量在某一维度上大小不同且都不是1，则报错；
3. 如果其中一个张量维度数更少，则在前面补1。

在bigram模型的归一化中，这一机制至关重要：

```python
counts = logits.exp()  # 形状: (228146, 27)
counts_sum = counts.sum(1, keepdim=True)  # 形状: (228146, 1)
probs = counts / counts_sum  # 广播: (228146, 27) / (228146, 1)
```

这里 `(228146, 1)` 的 `counts_sum` 被广播为 `(228146, 27)`——每一行被复制27次，然后逐元素相除。这样每一行都被自己的和归一化，得到正确的概率分布。

如果省略 `keepdim=True`，`counts.sum(1)` 的形状会是 `(228146,)`。此时 `(228146, 27) / (228146,)` 会触发广播，但它会尝试**按最后一个维度**对齐，导致按**列**归一化而非按行归一化！这个bug不会报错——PyTorch会静默执行，给出完全错误的结果。这是Karpathy在课程中反复强调的一个陷阱。

#### 2.4.3 Negative Log Likelihood（NLL）Loss：衡量模型预测与真实数据的差距

负对数似然（Negative Log-Likelihood, NLL）是语言模型中最核心的损失函数。它的推导始于**最大似然估计（Maximum Likelihood Estimation, MLE）**：我们希望找到模型参数，使得训练数据出现的概率最大。

给定训练集中的 $N$ 个bigram，数据的联合似然是：

$$\mathcal{L}(\theta) = \prod_{i=1}^{N} P(x_i^{(\text{next})} \mid x_i^{(\text{current})}; \theta)$$

直接最大化这个乘积在数值上很不稳定（大量小于1的数相乘会导致下溢）。取对数将其转化为求和：

$$\log \mathcal{L}(\theta) = \sum_{i=1}^{N} \log P(x_i^{(\text{next})} \mid x_i^{(\text{current})}; \theta)$$

由于优化器通常被设计为**最小化**损失，我们取负号，并除以 $N$ 取平均：

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i^{(\text{next})} \mid x_i^{(\text{current})}; \theta)$$

这就是**平均负对数似然**，也是**交叉熵损失（cross-entropy loss）**在离散分布情况下的表达。损失越小，模型对训练数据的预测越准确。理想情况下，如果模型对每个样本都给出100%的确定性预测（概率为1），则 $\log(1) = 0$，损失为0。

#### 2.4.4 使用F.cross_entropy简化：数值稳定性更好，计算更高效

我们在2.3.3节中手动实现了softmax + NLL，但这种方式存在数值稳定性问题。当logits的绝对值很大时（比如正值超过80），`.exp()` 会溢出为 `inf`；当所有logits都是很小的负数时，`.exp()` 会下溢为0，导致除以0的灾难。

`torch.nn.functional.cross_entropy` 将softmax和NLL融合为一个操作，并在内部进行了数值稳定优化：

```python
for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W

    # 直接用 cross_entropy，等价于 softmax + NLL，但更稳定
    loss = F.cross_entropy(logits, ys)

    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
```

`F.cross_entropy` 内部的数值稳定化技巧在于：它会对logits减去每行的最大值，使得所有参与指数运算的数值都非正，从而避免溢出：

$$\text{logits}_{\text{stable}} = \text{logits} - \max(\text{logits})$$

$$\text{loss} = -\text{logits}[y] + \max(\text{logits}) + \log \sum_j \exp(\text{logits}_{\text{stable}}[j])$$

这个被称为 **Log-Sum-Exp trick**，是现代深度学习框架中softmax交叉熵的标准实现方式。对于任何实际应用，都应该使用 `F.cross_entropy` 而非手动的softmax + log + mean组合——它更简洁、更稳定、通常也更快（ PyTorch可以对其进行算子融合优化）。

---

### 2.5 模型平滑与正则化

#### 2.5.1 模型平滑（model smoothing）：给计数矩阵加一个常数，防止概率为零

在2.2.3节中我们已经遇到了Laplace平滑：给计数矩阵的每个元素加1，避免任何bigram的概率为0。更一般地，我们可以加任意常数 $\alpha$：

$$\tilde{N}[i, j] = N[i, j] + \alpha$$

$\alpha$ 越大，概率分布越**均匀**（uniform）。极端情况下，$\alpha \to \infty$ 时所有bigram的概率趋于 $1/27$，模型完全"无知"。$\alpha = 0$ 时不对数据做任何修正，但会赋予未见过的bigram 0概率，导致对数似然无穷大。

$\alpha$ 的选择是一个超参数，它控制了模型的"保守程度"——较大的 $\alpha$ 让模型对新组合更宽容，但也削弱了它从数据中学到的真实规律。

#### 2.5.2 L2正则化与平滑的等价性：从损失函数角度理解平滑

这是本章最深刻的洞察之一：**计数法中的模型平滑，在神经网络中等价于L2正则化**。

L2正则化（也称权重衰减，weight decay）在损失函数中增加一项：

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{NLL}} + \lambda \cdot \frac{1}{n^2} \sum_{i,j} W_{i,j}^2$$

其中 $\lambda$ 控制正则化强度。当 $\lambda$ 较大时，优化器被"鼓励"让权重 $W_{i,j}$ 接近0。回忆2.3.4节的等价性——$W \approx \log(N + 1)$。如果 $W$ 被推向0，那么 $\exp(W)$ 被推向1，所有字符的logits趋于相等，softmax的输出趋于**均匀分布**。这正是模型平滑的效果！

更精确地看，考虑神经网络损失加上L2正则化：

$$\text{Loss} = -\frac{1}{N} \sum_i \log \frac{\exp(W[x_i, y_i])}{\sum_j \exp(W[x_i, j])} + \lambda \sum_{i,j} W_{i,j}^2$$

对 $W[a, b]$ 求梯度并令其为零（寻找最优解），可以发现当 $\lambda > 0$ 时，最优 $W$ 满足：

$$W^*[a, b] \approx \log(N[a, b] + 2\lambda \cdot n_b) - \log Z_a$$

其中 $Z_a$ 是归一化常数，$n_b$ 是某个与目标字符相关的项。这明确显示了正则化项 $\lambda$ 的作用类似于计数法中的平滑常数 $\alpha$——它在每个计数上"加"了一个由正则化强度决定的量。

```python
# 在训练中加入L2正则化
for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    loss = F.cross_entropy(logits, ys) + 0.01 * (W ** 2).mean()
    #                                         ^^^^^^^^^^^^^ L2正则化项
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
```

这个视角极其重要：它告诉我们，正则化不只是一个"防止过拟合的技巧"，它在统计上与**先验信念**（prior belief）紧密相连——L2正则化等价于假设权重服从高斯先验分布，这本质上是对模型复杂度的一种约束。

#### 2.5.3 训练集与验证集：评估模型泛化能力

在本章的bigram实验中，我们实际上一直在**训练集**上评估模型。bigram模型的参数量（$27 \times 27 = 729$）与训练数据量（约22.8万bigram）相比极小，过拟合的风险很低。但随着模型复杂度增加，划分训练集和验证集来评估**泛化能力**（generalization）变得至关重要。

一个模型在训练集上表现良好但无法生成合理的名字，说明它可能只是"死记硬背"了训练数据，而没有学到真正的语言规律。将数据集划分为训练集和验证集（通常80/20或90/10比例），在训练集上优化模型，在验证集上评估损失，是深度学习实践中的标准流程。后续章节中的MLP和Transformer模型将更加依赖这一评估框架。

---

### 2.6 课后练习与资源

#### 2.6.1 练习题

**练习1：广播机制的理解**

解释为什么以下两种归一化方式会产生不同的结果（提示：考虑PyTorch广播规则）：

```python
# 方式A
P1 = N.float()
P1 /= P1.sum(1, keepdim=True)

# 方式B
P2 = N.float()
P2 /= P2.sum(1)
```

绘制 $P1[0]$ 和 $P2[0]$ 的柱状图，观察它们的差异。

**练习2：模型平滑参数实验**

在计数法的平滑中，尝试不同的 $\alpha$ 值（如0, 1, 10, 100），分别计算训练集上的NLL损失，并观察生成名字的质量变化。$\alpha$ 多大时生成的名字开始变得完全随机？

**练习3：L2正则化的等价验证**

在神经网络训练中，尝试不同的L2正则化强度（如 `0.0, 0.01, 0.1, 1.0, 10.0`），记录最终损失。然后计算 $\exp(W)$ 矩阵，观察它是否与 $(N + \alpha)$ 的归一化版本相似——正则化强度是否与等效的平滑常数 $\alpha$ 存在对应关系？

**练习4：Softmax数值稳定性**

构造一组极端的logits（如 `[100.0, 0.0, -100.0]`），分别用手动softmax和 `F.cross_entropy` 计算损失和梯度。观察哪种方式会产生数值溢出？阅读 `F.cross_entropy` 的文档，理解其内部的数值稳定化实现。

**练习5：从神经网络模型采样**

修改2.2.4节的采样代码，使其从训练好的**神经网络模型**而非计数矩阵中采样生成新名字。注意你需要在前向传播中使用 `torch.no_grad()` 或者在推理模式下执行，避免不必要的梯度计算。

**练习6：比较两种方法的损失**

精确比较计数法和神经网络法（无正则化）的最终损失。它们是否收敛到完全相同的值？如果有微小差异，思考可能的原因（提示：考虑优化过程的随机性和收敛精度）。

**练习7：One-hot编码的效率**

one-hot编码会创建一个 $(228146, 27)$ 的矩阵，其中绝大多数元素是0。研究PyTorch的 `torch.nn.Embedding` 层，它如何更高效地实现"从one-hot向量到权重矩阵行的选取"操作？修改训练代码使用 `nn.Embedding` 替代 `F.one_hot`。

**练习8：多字符名字的损失分解**

挑选几个特定的训练名字（如 `"emma"` 和 `"john"`），分别计算它们在计数模型下的对数似然（即每个bigram概率的对数之和）。哪个名字更容易被模型"解释"？这与名字的常见程度是否一致？

#### 2.6.2 推荐学习资源

**核心资源**

- [YouTube视频 - building makemore (bigram)](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy的原版课程视频[^2^]
- [Jupyter Notebook - makemore_part1_bigrams.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb) — 课程配套的完整代码[^3^]
- [makemore GitHub仓库](https://github.com/karpathy/makemore) — 项目主页和names.txt数据集[^4^]

**练习题与社区资源**

- [makemore bigram 练习题](https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/Exercises/makemore%20exercises/part1_bigrams._exercises.ipynb) — 由社区整理的配套练习[^5^]
- [nn-zero-to-hero-notes](https://github.com/MK2112/nn-zero-to-hero-notes) — 社区详细笔记和解答[^6^]
- [Anri-Lombard/makemore](https://github.com/Anri-Lombard/makemore) — 获得Karpathy本人点赞的社区笔记[^7^]

**经典论文**

- [Bengio et al. 2003 - A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — 神经网络语言模型的开创性工作，提出了用分布式表示（distributed representations）替代稀疏计数的核心思想，为后续所有神经语言模型奠定了基础[^8^]
- [Mikolov et al. 2013 - Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Word2Vec论文，展示了如何用浅层神经网络学习高质量的词向量[^9^]
- [Vaswani et al. 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer架构论文，提出了自注意力机制，是现代大语言模型的基石[^10^]

**后续课程**

| 课程 | 主题 | YouTube链接 |
|---|---|---|
| 第3课 | MLP多层感知机 | [TCH_1BHY58I](https://youtu.be/TCH_1BHY58I)[^11^] |
| 第4课 | 激活函数、梯度与BatchNorm | [P6sfmUTpUmc](https://www.youtube.com/watch?v=P6sfmUTpUmc)[^12^] |
| 第5课 | 成为反向传播忍者 | [q8SA3rM6ckI](https://www.youtube.com/watch?v=q8SA3rM6ckI)[^13^] |
| 第6课 | WaveNet | [t3YJ5hKiMQ0](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)[^14^] |
| 第7课 | 从头构建GPT | [kCc8FmEb1nI](https://www.youtube.com/watch?v=kCc8FmEb1nI)[^15^] |

**深度学习基础参考**

- [PyTorch广播机制教程](https://pytorch.org/docs/stable/notes/broadcasting.html) — 官方文档，强烈推荐仔细阅读以避免广播bug[^16^]
- [PyTorch CrossEntropyLoss文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — 理解交叉熵损失的精确计算方式[^17^]
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) — 自然语言处理经典教材的第3章和第4章涵盖了n-gram语言模型的完整理论[^18^]

---

**参考文献标注说明**

[^1^]: names.txt数据集，来源于美国社会保障局2018年公开数据，`https://github.com/karpathy/makemore/blob/master/names.txt`

[^2^]: Karpathy, A. (2022). *The spelled-out intro to language modeling: building makemore*. YouTube, `https://www.youtube.com/watch?v=PaCmpygFfXo`

[^3^]: Karpathy, A. (2022). *makemore_part1_bigrams.ipynb*, `https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb`

[^4^]: Karpathy, A. (2022). *makemore GitHub repository*, `https://github.com/karpathy/makemore`

[^5^]: Community exercises for makemore Part 1, `https://github.com/0ssamaak0/Karpathy-Neural-Networks-Zero-to-Hero/blob/master/Exercises/makemore%20exercises/part1_bigrams._exercises.ipynb`

[^6^]: MK2112. *nn-zero-to-hero-notes*, `https://github.com/MK2112/nn-zero-to-hero-notes`

[^7^]: Lombard, A. *makemore community notes*, `https://github.com/Anri-Lombard/makemore`

[^8^]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3(Feb), 1137-1155. `https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf`

[^9^]: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*. `https://arxiv.org/abs/1301.3781`

[^10^]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*. `https://arxiv.org/abs/1706.03762`

[^11^]: Karpathy, A. (2022). *Building makemore Part 2: MLP*. YouTube, `https://youtu.be/TCH_1BHY58I`

[^12^]: Karpathy, A. (2022). *Building makemore Part 3: Activations & Gradients, BatchNorm*. YouTube, `https://www.youtube.com/watch?v=P6sfmUTpUmc`

[^13^]: Karpathy, A. (2022). *Building makemore Part 4: Becoming a Backprop Ninja*. YouTube, `https://www.youtube.com/watch?v=q8SA3rM6ckI`

[^14^]: Karpathy, A. (2022). *Building makemore Part 5: WaveNet*. YouTube, `https://www.youtube.com/watch?v=t3YJ5hKiMQ0`

[^15^]: Karpathy, A. (2022). *Let's build GPT from scratch*. YouTube, `https://www.youtube.com/watch?v=kCc8FmEb1nI`

[^16^]: PyTorch Broadcasting Semantics, `https://pytorch.org/docs/stable/notes/broadcasting.html`

[^17^]: PyTorch CrossEntropyLoss documentation, `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`

[^18^]: Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed.), `https://web.stanford.edu/~jurafsky/slp3/`



---

## 3. 第3课：makemore Part 2 — 多层感知机(MLP)

在前一课中，我们实现了bigram语言模型——它根据前一个字符来预测下一个字符。这个模型虽然简单直观，但其能力也受限于仅看一个字符的局限性。自然语言中存在着大量跨越多个字符的依赖关系：比如英语中"qu"几乎总是连在一起出现，某些前缀如"un-"、"pre-"会系统性地影响后续字符的选择概率。要捕捉这些模式，模型需要"记住"更长的上下文。

这一课的核心目标是构建一个多层感知机(Multi-Layer Perceptron, MLP)字符级语言模型。我们将从Bengio等人2003年的经典论文出发，理解为什么神经网络加上分布式表示(distributed representation)能够克服维度诅咒(curse of dimensionality)，然后逐步实现完整的MLP架构、训练循环，并学习机器学习中的基础实践：学习率调优、数据集划分、过拟合与欠拟合的判断。本课的代码实现参考了Karpathy的GitHub Notebook[^1^]，而模型架构则源自Bengio et al. 2003的开创性工作[^2^]。

### 3.1 从Bigram到上下文模型

#### 3.1.1 Bigram的局限

Bigram模型只考虑前一个字符来预测下一个字符。用概率的语言来表达，它建模的是 $P(x_t | x_{t-1})$，即在给定前一个字符的条件下下一个字符的概率。这种条件独立性假设虽然让模型简单高效，但也严重限制了它的表达能力。

举个例子，假设我们训练数据中有名字"emily"和"emma"。当模型看到序列".em"时，bigram模型只看最后一个字符"m"来决定下一个字符。但实际上，前缀".e"的信息对于预测后续字符也非常有价值——以".e"开头的名字往往遵循特定的模式。更极端的例子是：某些语言现象需要更长的上下文才能判断，比如一个名字是否即将结束（看到两个连续元音后结束的概率）。

从信息论的角度来思考，bigram模型能够利用的信息量非常有限。如果我们将语言建模的目标理解为压缩——即用最少的比特数来编码一个文本序列——那么bigram模型只能利用局部统计规律，对于更长距离的依赖关系则无能为力。

#### 3.1.2 扩展上下文窗口

自然的想法是：如果模型能看到前几个字符，而不是只看前一个，它的预测能力会不会大幅提升？这就是上下文模型(context model)的核心思想。我们将使用一个固定长度的窗口——称为block_size或context window——来收集前block_size个字符，然后用这些字符的信息来预测下一个。

具体来说，如果block_size = 3，那么对于名字".emma."，我们会构造如下训练样本：

| 输入 (context) | 输出 (target) |
|:---:|:---:|
| [0, 0, 0] ('...') | 5 ('e') |
| [0, 0, 5] ('..e') | 13 ('m') |
| [0, 5, 13] ('.em') | 13 ('m') |
| [5, 13, 13] ('emm') | 1 ('a') |
| [13, 13, 1] ('mma') | 0 ('.') |

这里的0是特殊标记'.'的索引，表示字符串的开始或结束。每个上下文是一个长度为3的整数数组，对应字符在词汇表中的索引。目标则是紧随其后的那个字符的索引。

构建训练集的代码如下：

```python
block_size = 3  # context length: how many characters do we take 
                # to predict the next one?

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
```

这段代码展示了滑动窗口的核心逻辑。`context = [0] * block_size`初始化为全零（对应'.'字符），然后对每个单词中的每个字符，我们将当前上下文作为输入X，当前字符作为目标Y，然后通过`context = context[1:] + [ix]`将窗口向右滑动一位：去掉最左边的一个字符，在最右边追加新字符。这个操作模拟了序列在时间轴上的推进。

#### 3.1.3 Bengio 2003论文的核心思想

Yoshua Bengio等人在2003年发表的论文"A Neural Probabilistic Language Model"[^2^]提出了一个革命性的观点：用神经网络学习词的分布式表示(distributed representation)，并以此来建模语言序列的概率分布。

论文的核心洞察围绕**维度诅咒(curse of dimensionality)**展开。假设词汇表有V个词，我们要建模一个n-gram序列的概率。可能的序列组合数是 $V^n$，这会随着n呈指数级增长。对于任何实际大小的语料库，都不可能覆盖所有可能的n-gram组合。传统的n-gram模型通过平滑(smoothing)和回退(backoff)等技巧来缓解这个问题，但本质上是"死记硬背"——它只能在训练数据中见过的短片段上进行插值。

Bengio等人提出了一个根本不同的解决思路：**分布式表示**。每个词被表示为一个低维的连续向量（维度m远小于词汇表大小V）。语义相似的词在这个向量空间中会自动靠近。关键优势在于：如果"dog"和"cat"有相似的向量表示，且"the"和"a"也有相似的表示，那么模型在训练数据中看到了"The cat is walking in the bedroom"之后，它不仅能评估这个句子的概率，还能自然地泛化到"A dog was running in a room"——即使这个具体组合从未在训练数据中出现。这是因为概率函数是这些连续向量表示的平滑函数，相似的输入向量会产生相似的输出概率。

论文提出的模型架构与我们这一课要实现的几乎完全一致：

1. 输入是前n-1个词的索引
2. 通过嵌入矩阵C将这些索引映射为m维的特征向量
3. 将n-1个嵌入向量拼接成一个长向量
4. 通过一个带tanh激活函数的隐藏层
5. 最后通过softmax输出层得到下一个词的概率分布

这个架构的两个部分——词的特征向量和概率函数——是**联合学习**(jointly learned)的，通过最大化训练数据的似然来端到端地优化。在当时，这是一个非常大胆的方法，因为模型可能有数百万个参数，训练这样的模型本身就是一个重大挑战。但实验结果表明，这种方法显著超越了当时的最优n-gram模型。

### 3.2 MLP架构设计

现在我们从理论走向实践，逐步实现Bengio 2003模型架构的每一个组件。我们的实现使用字符而非词作为基本单元（字符级语言模型），但核心原理完全相同。

#### 3.2.1 Embedding层

Embedding层是连接离散符号世界与连续向量空间的桥梁。在我们的名字数据集中，有27个可能的字符（26个小写字母加'.'），每个字符用一个整数索引表示。Embedding层将这些整数索引映射为稠密的、可学习的向量。

实现上，Embedding层就是一个形状为 `(vocab_size, embedding_dim)` 的参数矩阵C。当输入一个字符索引i时，我们取出C的第i行作为该字符的嵌入向量。在PyTorch中，这个操作可以通过整数索引直接完成，这种机制称为**索引查找**(index lookup)或**查表**(lookup table)。

```python
# 参数初始化
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)  # embedding table: 27 chars, 10-dim each
```

这里我们创建了一个27×10的矩阵C，每行对应一个字符的10维嵌入向量。初始化使用标准正态分布。当我们有多个输入时，PyTorch支持非常优雅的**多维索引**：

```python
# X is (batch_size, block_size) containing character indices
emb = C[X]  # (batch_size, block_size, 10)
```

如果X的形状是 `(batch_size, 3)`，即每个样本包含3个字符索引，那么`C[X]`的形状就是 `(batch_size, 3, 10)`。PyTorch会自动为X中的每个元素查找对应的嵌入向量，保持原始维度结构。这比手动构造one-hot向量再做矩阵乘法高效得多——本质上，one-hot向量与矩阵的乘法等价于从矩阵中选取对应行，而直接索引避免了不必要的计算。

从数学上看，Embedding层与线性层有着密切的联系。如果我们用一个one-hot向量e_i（第i位为1，其余为0）去乘矩阵C，结果恰好是C的第i行。所以Embedding本质上是一个"稀疏版"的线性变换：输入是一个整数索引（隐含了一个one-hot表示），输出是对应的嵌入向量。

#### 3.2.2 上下文Embedding拼接

模型需要同时利用block_size个字符的信息来做出预测。在Bengio的架构中，这是通过将多个字符的嵌入向量**拼接**(concatenate)成一个长向量来实现的。

```python
# emb has shape (batch_size, 3, 10)
# we concatenate the 3 embedding vectors into one long vector
emb_cat = emb.view(-1, 30)  # (batch_size, 30)
```

`view(-1, 30)`是PyTorch中的张量重塑操作。`-1`表示该维度的大小由PyTorch自动推断（即batch_size），30则是3个10维嵌入向量的拼接长度（3 × 10 = 30）。这行代码之所以有效，是因为PyTorch的张量在内存中是连续存储的一维数组，而`view`只是改变了对这个数组的"解读方式"——不改变底层数据，只改变shape和stride。这一点我们稍后在讨论PyTorch内部机制时会深入理解。

拼接操作的意义在于：它将上下文中的所有字符信息汇集到一个向量中，为后续的神经网络层提供完整的输入表示。注意这里没有做任何复杂的特征交互——只是简单地将向量首尾相接。真正的特征学习和非线性变换将交给后续的隐藏层来完成。

#### 3.2.3 隐藏层

隐藏层是MLP引入非线性表达能力的关键。如果没有隐藏层和激活函数，无论堆叠多少线性变换，最终的输出仍然是输入的线性函数——模型的表达能力将与简单的线性回归相当。

```python
W1 = torch.randn((30, 200), generator=g)  # hidden layer weights
b1 = torch.randn(200, generator=g)        # hidden layer bias

# forward pass through hidden layer
h = torch.tanh(emb_cat @ W1 + b1)  # (batch_size, 200)
```

这里W1的形状是 `(30, 200)`：30来自输入向量的维度（3个字符 × 10维嵌入），200是隐藏层的神经元数量——这是一个**超参数**(hyperparameter)，我们可以根据需要调整。`emb_cat @ W1`是矩阵乘法，加上偏置b1后，再通过tanh激活函数。

**tanh**(hyperbolic tangent)是一个S型激活函数，将任意实数映射到 (-1, 1) 区间：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

tanh的导数是 $1 - \tanh^2(x)$，当x接近0时导数接近1，当|x|很大时导数接近0。这意味着当神经元的预激活值(pre-activation)处于合理范围时，梯度可以顺畅地反向传播；但当预激活值过大或过小时，神经元会进入**饱和区**(saturation region)，梯度几乎为零，导致学习停滞。这个问题在训练深层网络时尤为突出，我们会在下一课中详细讨论。

#### 3.2.4 输出层与交叉熵损失

输出层将隐藏层的表示转换为每个可能字符的"分数"(logits)，然后通过softmax转换为概率分布。

```python
W2 = torch.randn((200, 27), generator=g)  # output layer weights
b2 = torch.randn(27, generator=g)         # output layer bias

# forward pass through output layer
logits = h @ W2 + b2  # (batch_size, 27)
```

logits的形状是 `(batch_size, 27)`：每一行对应一个训练样本，27个值分别表示27个可能字符的"原始分数"。分数越高，表示模型认为该字符越可能是下一个字符。

要将logits转换为概率分布，理论上需要经过softmax：

$$P(i) = \frac{e^{\text{logits}_i}}{\sum_j e^{\text{logits}_j}}$$

然后计算**负对数似然损失**(negative log-likelihood loss)：

$$\text{loss} = -\frac{1}{N} \sum_k \log P(y_k)$$

其中 $y_k$ 是第k个样本的真实标签（即正确的下一个字符的索引）。但在PyTorch中，我们直接使用`F.cross_entropy`来完成这一切：

```python
import torch.nn.functional as F

loss = F.cross_entropy(logits, Y)  # Y is (batch_size,) containing target indices
```

`F.cross_entropy`在内部高效地实现了softmax + log + negative log-likelihood的组合，而且数值稳定性更好（它会在内部减去最大值来防止指数爆炸）。更重要的是，与手动实现相比，`F.cross_entropy`避免了创建大量的中间张量（如概率矩阵），使得前向传播和反向传播都更加高效。

整个模型的前向传播流程可以总结为：

```
Input indices:       (batch_size, 3)
  |
  v
Embedding lookup:    (batch_size, 3, 10)   [C[X]]
  |
  v
Flatten/Concat:      (batch_size, 30)      [view]
  |
  v
Hidden Layer:        (batch_size, 200)     [tanh(x @ W1 + b1)]
  |
  v
Output Layer:        (batch_size, 27)      [logits = h @ W2 + b2]
  |
  v
Cross Entropy Loss:  scalar                [F.cross_entropy]
```

### 3.3 训练基础

有了模型架构，下一步是训练。训练神经网络本质上是一个优化问题：我们希望通过调整模型的参数（C, W1, b1, W2, b2），使得在训练数据上的平均损失最小化。这一节我们将覆盖训练的核心环节。

#### 3.3.1 训练集、验证集、测试集的划分

在机器学习中，我们关心的不是模型在训练数据上表现多好，而是它在新数据上的**泛化能力**(generalization ability)——即模型能否对从未见过的输入做出准确预测。为了准确评估这一点，我们需要将数据划分为三个互不相交的集合。

```python
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr,  Ytr  = build_dataset(words[:n1])      # 80% training
Xdev, Ydev = build_dataset(words[n1:n2])    # 10% dev/validation
Xte,  Yte  = build_dataset(words[n2:])      # 10% test
```

三个集合各有明确的用途：

**训练集 (Training Set, 80%)**：用于模型参数的更新。梯度下降只在这个集合上计算梯度和调整参数。

**验证集 / 开发集 (Validation / Dev Set, 10%)**：用于超参数调优(hyperparameter tuning)和模型选择。我们尝试不同的超参数组合（如学习率、隐藏层大小、嵌入维度等），在验证集上评估每种组合的表现，选择表现最好的那一组。注意，模型**从未**在验证集上直接训练，所以验证集上的损失反映了模型对未见过数据的泛化能力。

**测试集 (Test Set, 10%)**：仅在最终评估时使用，反映模型的真实性能。一个至关重要的原则是：**绝对不能使用测试集来调参**。一旦你用测试集的结果来指导模型改进（无论是调超参数还是选模型架构），测试集就不再是"未见过的数据"了——模型间接地"学习"了测试集的信息。这会导致对模型性能的过于乐观的估计。测试集应该被当作一个神圣的、仅在最后使用一次的评估标准。

这种划分在工业界和学术界都是标准做法。对于我们的名字数据集（约32000个名字），80/10/10的划分提供了充足的训练数据和可靠的评估基础。

#### 3.3.2 学习率调优

学习率(learning rate, lr)是梯度下降中最关键的超参数之一。它控制了每次参数更新的步长：

```python
# parameter update step
p.data += -lr * p.grad
```

学习率太小，模型收敛极其缓慢，可能需要数百万次迭代才能达到较好的损失值。学习率太大，参数更新步长过大，可能直接"跳过"损失函数的最低点，在两侧震荡甚至发散（loss变为NaN）。

Karpathy在课程中介绍了一种非常实用的学习率搜索方法：系统地尝试多个数量级的学习率，观察哪种效果最好。

```python
lre = torch.linspace(-3, 0, 1000)  # 1000 points from -3 to 0
lrs = 10 ** lre                     # learning rates from 10^-3 to 10^0

lri = []
lossi = []

for i in range(1000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # update with current learning rate
    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    lri.append(lre[i])
    lossi.append(loss.item())
```

这段代码尝试从 $10^{-3}$ 到 $10^0$（即0.001到1.0）之间的1000个学习率值。每次迭代使用一个minibatch（32个样本）来快速评估该学习率的效果。训练结束后，我们可以绘制学习率与损失的关系图：

```python
plt.plot(lri, lossi)
```

通过这个图，我们可以观察到损失随学习率变化的趋势。通常，图的最左边（小学习率）损失下降缓慢，中间某个区域损失快速下降，而最右边（大学习率）损失开始震荡甚至飙升。一个好的学习率位于"快速下降区"的中间偏左位置——既足够大以确保快速收敛，又不至于太大而导致不稳定。对于我们的模型，0.1通常是一个不错的选择。

#### 3.3.3 学习率衰减

在训练的后期，当损失已经下降到接近最优值时，较大的学习率可能导致模型在最优点附近震荡而无法精确收敛。这时引入**学习率衰减**(learning rate decay)策略：逐步降低学习率，让模型以更精细的步长进行优化。

```python
for i in range(200000):
    # ... forward and backward pass ...
    
    # step learning rate decay
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

这里采用了一种简单的**阶梯衰减**(step decay)策略：前100000步使用学习率0.1，之后降低到0.01。这种策略在课程中被证明非常有效——先用较大的学习率快速接近最优区域，再用较小的学习率精细调整。

#### 3.3.4 训练循环的完整实现

将以上所有组件整合在一起，完整的训练循环如下：

```python
# parameter initialization
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10),   generator=g)
W1 = torch.randn((30, 200),  generator=g)
b1 = torch.randn(200,        generator=g)
W2 = torch.randn((200, 27),  generator=g)
b2 = torch.randn(27,         generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# training loop
for i in range(200000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    # forward pass
    emb = C[Xtr[ix]]                          # (32, 3, 10)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
    logits = h @ W2 + b2                       # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])    # scalar
    
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())
```

这个训练循环体现了深度学习训练的核心模式。首先，**minibatch构造**：从训练集中随机采样32个样本。使用minibatch而非整个数据集（full-batch）的原因在于效率——每次迭代只需要处理一小部分数据，使得参数更新更加频繁。虽然单个minibatch的梯度方向不如全数据集精确，但频繁更新的累积效果往往更好，而且计算上更加可行。随机采样的另一个好处是为训练引入了噪声，这有助于模型跳出局部最优和尖锐的极小值点。

第二步是**前向传播**(forward pass)：从输入到输出，依次计算每层的激活值和最终的损失。第三步是**反向传播**(backward pass)：调用`loss.backward()`，PyTorch自动计算损失对每个参数的梯度。这里需要先将所有参数的梯度清零（`p.grad = None`），因为PyTorch默认会累加梯度。最后一步是**参数更新**：沿梯度的反方向（即损失下降最快的方向）调整参数。

### 3.4 过拟合与欠拟合

训练过程中需要密切关注两个关键指标：训练集上的损失和验证集上的损失。这两个指标的相对关系揭示了模型是否处于健康的状态。

#### 3.4.1 欠拟合

**欠拟合(underfitting)**发生在模型容量(capacity)不足的情况下——模型太简单，无法捕捉数据中的复杂模式。判断欠拟合的信号是：训练损失和验证损失都很高，而且两者之间的差距不大。

在我们的名字数据集上，如果训练损失约为2.5，验证损失也约为2.5（而我们的bigram基线大约是2.4），这表明MLP模型并没有比简单的bigram模型学到更多东西。可能的原因包括：嵌入维度太小（比如只有2维，无法表达丰富的字符语义）、隐藏层神经元太少（比如只有10个，无法学习复杂的非线性映射）、或者训练迭代次数不够。

解决欠拟合的方法是**增加模型容量**：增大嵌入维度、增加隐藏层神经元数量、增加隐藏层的层数、或者增加训练时间。但要注意，容量增加也会带来过拟合的风险。

#### 3.4.2 过拟合

**过拟合(overfitting)**则发生在模型容量过大的情况下——模型不仅学习了数据中的一般规律，还"死记硬背"了训练数据中的噪声和特殊性。判断过拟合的信号是：训练损失很低，但验证损失明显高于训练损失。

在我们的训练中，如果在训练后期观察到训练损失降至1.8，而验证损失停留在2.3，这就是典型的过拟合信号。模型在训练数据上表现完美，但面对新数据时性能显著下降。它可能记住了训练集中特定名字的模式，而不是学习到了通用的命名规律。

#### 3.4.3 容量控制与超参数选择

在欠拟合和过拟合之间找到一个平衡点，是机器学习实践的核心挑战。以下超参数直接影响模型的容量：

| 超参数 | 对容量的影响 | 过小的问题 | 过大的问题 |
|:---:|:---:|:---:|:---:|
| embedding_dim | 字符表示的丰富度 | 无法区分不同字符的语义 | 参数量增加，可能过拟合 |
| n_hidden (隐藏层大小) | 模型的非线性表达能力 | 无法学习复杂模式 | 参数量剧增，容易过拟合 |
| block_size (上下文长度) | 能利用的历史信息量 | 忽略长距离依赖 | 输入维度增大，需要更多数据 |

除了这些结构超参数，**正则化**(regularization)也是控制过拟合的重要手段。在训练损失中加入权重衰减(weight decay)项：

$$\text{loss}_{\text{total}} = \text{loss}_{\text{CE}} + \lambda \sum_p p^2$$

其中λ是正则化强度。这个额外的惩罚项鼓励参数保持较小的值，防止模型过度依赖任何单个参数，从而提升泛化能力。

#### 3.4.4 通过验证集选择最佳超参数

实际调参的流程是一个迭代过程。我们使用验证集来指导超参数的选择，而不是训练集：

```python
@torch.no_grad()  # disable gradient tracking for evaluation
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val':   (Xdev, Ydev),
        'test':  (Xte, Yte),
    }[split]
    emb = C[x]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

注意这里使用了`@torch.no_grad()`装饰器来禁用梯度追踪——评估阶段不需要计算梯度，这可以节省内存和计算。

典型的调参策略是：先固定一组超参数，训练模型，记录验证损失；然后系统地改变一个超参数（如隐藏层大小从100到200到300），重复训练和评估；最后选择验证损失最低的那组超参数。这个过程只在验证集上进行。当找到最佳超参数后，我们才在测试集上做一次最终评估，得到模型的真实性能指标。

### 3.5 PyTorch Internals与高效实现

到目前为止，我们都是手动创建和管理每个参数张量。这在教学上很清晰，但在实践中会非常繁琐，尤其是对于深层网络。PyTorch提供了`torch.nn.Module`框架来标准化神经网络的构建方式。

#### 3.5.1 torch.nn.Module

`nn.Module`是PyTorch中所有神经网络模块的基类。它提供了统一的方式来定义网络结构、管理参数、以及实现前向传播逻辑。核心思想是将模型封装为一个类，参数作为类的属性，前向传播逻辑在`forward`方法中定义。

```python
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, n_hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * block_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)
    
    def forward(self, x):
        # x: (batch_size, block_size)
        emb = self.emb(x)              # (batch_size, block_size, embedding_dim)
        emb = emb.view(x.shape[0], -1)  # (batch_size, block_size * embedding_dim)
        h = torch.tanh(self.fc1(emb))  # (batch_size, n_hidden)
        logits = self.fc2(h)            # (batch_size, vocab_size)
        return logits
```

这个重构后的模型与之前的手动实现完全等价，但代码更加清晰和模块化。`nn.Module`自动处理参数的注册和管理——我们不再需要手动创建参数列表，也不需要逐个启用`requires_grad`。

#### 3.5.2 参数管理：nn.Parameter与parameters()

`nn.Module`的强大之处在于它的参数管理机制。当我们将`nn.Embedding`、`nn.Linear`等模块作为类的属性赋值时，`nn.Module`会自动识别这些模块中的可学习参数，并通过`parameters()`方法统一返回。

```python
model = MLPModel(vocab_size=27, embedding_dim=10, 
                 block_size=3, n_hidden=200)

# all learnable parameters
for p in model.parameters():
    print(p.shape)

# total parameter count
print(sum(p.numel() for p in model.parameters()))
```

`parameters()`方法递归地遍历模型中的所有子模块，收集所有的`nn.Parameter`对象。这使得训练循环中的参数更新变得异常简洁：

```python
# training loop with nn.Module
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(200000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这里我们进一步使用了`torch.optim.SGD`优化器来处理参数更新。`optimizer.zero_grad()`清零所有参数的梯度，`optimizer.step()`执行参数更新。这使得训练循环更加简洁，同时也为使用更复杂的优化器（如Adam、RMSprop等）提供了统一的接口。

#### 3.5.3 使用标准模块重构代码

让我们对比手动实现和使用`nn.Module`的实现：

**手动实现**：

```python
# manual parameter creation and management
C  = torch.randn((27, 10),   generator=g)
W1 = torch.randn((30, 200),  generator=g)
b1 = torch.randn(200,        generator=g)
W2 = torch.randn((200, 27),  generator=g)
b2 = torch.randn(27,         generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# manual forward pass
emb = C[X]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)

# manual backward and update
for p in parameters:
    p.grad = None
loss.backward()
for p in parameters:
    p.data += -lr * p.grad
```

**nn.Module实现**：

```python
# model definition
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 10)
        self.fc1 = nn.Linear(30, 200)
        self.fc2 = nn.Linear(200, 27)
    
    def forward(self, x):
        x = self.emb(x).view(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

model = MLPModel()

# clean forward pass
logits = model(X)
loss = F.cross_entropy(logits, Y)

# clean backward and update
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

重构后的代码有几个显著优势。首先是**可读性**：模型的结构在`__init__`中一目了然，前向传播逻辑在`forward`中清晰表达。其次是**可维护性**：添加新的层只需要在`__init__`中增加一行，在`forward`中增加对应的调用即可。最后是**可扩展性**：`nn.Module`生态系统包含了卷积层、循环层、归一化层、 dropout等大量预建模块，可以方便地组合构建复杂架构。

#### 3.5.4 PyTorch Tensor的内部机制

Karpathy在课程中深入讲解了PyTorch Tensor的底层实现，这对于理解高效神经网络实现至关重要。

PyTorch的`torch.Tensor`由两个核心部分组成：**Storage**和**View**。Storage是一个一维的连续数组，存储了张量的实际数据。而View则通过shape、stride和storage_offset来描述如何从这个一维数组中解读出多维结构。

例如，一个形状为 `(3, 4)` 的张量，其底层Storage是一个长度为12的一维数组。stride告诉我们沿每个维度移动一步需要跳过多少个元素：对于这个张量，stride是 `(4, 1)`，意味着沿第0维移动一步跳过4个元素，沿第1维移动一步跳过1个元素。

`view`操作之所以高效，就是因为它不改变底层的Storage，只是改变了解读这个Storage的方式。比如将一个 `(32, 3, 10)` 的张量`view`为 `(32, 30)`，底层的数据完全不动，只是改变了shape和stride。这与`reshape`（在数据不连续时可能需要数据拷贝）形成对比。

理解这些内部机制对于写出高效的神经网络代码非常重要。比如，不必要的`transpose`、`permute`和`contiguous`调用会增加内存拷贝的开销。在性能敏感的场合，保持对Tensor内存布局的意识能够显著提升训练速度。Edward Yang的PyTorch Internals博客[^3^]是深入了解这些机制的优秀资源。

### 3.6 课后练习与资源

#### 3.6.1 练习题

以下练习改编自Karpathy课程视频描述中的E01-E03[^4^]，旨在帮助巩固本课的核心概念。

**E01：超参数搜索**

调整模型的超参数（embedding维度、隐藏层大小、学习率、训练迭代次数等），尝试使验证集上的损失低于2.2。建议你采用系统化的方法：每次只改变一个超参数，记录验证损失的变化，逐步逼近最优配置。思考一下：哪些超参数对性能的影响最大？增大隐藏层是否总能带来更好的性能？在什么情况下验证损失会开始上升（过拟合的信号）？

**E02：权重初始化分析**

本课中的代码在权重初始化上并不精细。这个问题引导你深入理解初始化的重要性。

(1) 理论分析：如果模型在初始化时输出完全均匀的概率分布（即每个字符的概率都是1/27），那么交叉熵损失应该是多少？计算这个理论值。然后运行代码，观察实际的初始损失。两者是否接近？如果差距很大，思考为什么会这样。（提示：考虑权重初始化的随机性对logits分布的影响。）

(2) 实践调整：能否通过调整初始化策略，使初始损失接近理论值？具体而言，尝试将最后一层的权重W2乘以较小的值（如0.01），将偏置b2初始化为0。再次观察初始损失。为什么这种调整能让初始损失更接近理论值？（提示：较小的权重意味着较小的logits差异，softmax输出更趋于均匀。）

**E03：阅读Bengio 2003论文**

仔细阅读Bengio等人2003年的论文"A Neural Probabilistic Language Model"[^2^]。从论文中选择一个你感兴趣的想法（例如：直接使用字符而非词作为基本单元、不同的网络架构变体、正则化方法、或者对嵌入矩阵的初始化策略），在你的名字数据集上实现它。记录你的实现细节和实验结果——它是否提升了验证损失？为什么有效或无效？

#### 3.6.2 推荐学习资源

| 资源 | 链接 | 说明 |
|:---:|:---:|:---|
| **课程视频** | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) | Karpathy原视频，时长约75分钟[^4^] |
| **课程Notebook** | [GitHub](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) | 本课完整代码实现[^1^] |
| **Colab版本** | [Colab](https://colab.research.google.com/drive/1YIfmkftLrz6MPTOO9Vwqrop2Q5llHIGK?usp=sharing) | 可直接运行的Jupyter Notebook |
| **Bengio 2003论文** | [PDF](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) | MLP语言模型的开创性工作[^2^] |
| **makemore项目** | [GitHub](https://github.com/karpathy/makemore) | 完整项目仓库 |
| **PyTorch Internals** | [Blog](http://blog.ezyang.com/2019/05/pytorch-internals/) | PyTorch张量内部机制详解[^3^] |
| **系列播放列表** | [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Neural Networks: Zero to Hero全集 |
| **PyTorch nn教程** | [官方文档](https://pytorch.org/tutorials/beginner/nn_tutorial.html) | 从零开始理解nn.Module[^5^] |

**阅读建议**：Bengio 2003的论文虽然发表于二十年前，但其中关于分布式表示和维度诅咒的讨论至今仍然深刻 relevant。建议至少精读论文的前三节（Introduction、The Model、Training），理解作者如何通过连续向量表示来克服离散符号空间的指数爆炸问题。PyTorch Internals博客则适合对底层实现感兴趣的读者，它详细解释了Storage、View和Autograd的内部工作原理。

---

**本章注释**：

[^1^]: Karpathy, A. "makemore part 2: MLP" Jupyter Notebook. https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb

[^2^]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. "A Neural Probabilistic Language Model." Journal of Machine Learning Research, 3(Feb):1137-1155, 2003. https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

[^3^]: Yang, E. "PyTorch Internals." http://blog.ezyang.com/2019/05/pytorch-internals/

[^4^]: Karpathy, A. "Building makemore Part 2: MLP." YouTube, 2022. https://www.youtube.com/watch?v=TCH_1BHY58I

[^5^]: PyTorch. "What is torch.nn really?" PyTorch Tutorials. https://pytorch.org/tutorials/beginner/nn_tutorial.html



---

# 4. 第4课：makemore Part 3 — 激活函数、梯度流与BatchNorm

在上一课中，我们搭建了一个三层MLP来预测下一个字符，训练loss收敛到了约2.2。但那个网络只有一层隐藏层。一个自然的想法是：如果我们加深网络——堆叠五层、六层甚至更多——模型应该能学到更复杂的模式，表现也会更好。可现实往往是残酷的。当你真的把网络加深，loss可能不会下降，反而出现震荡、发散，或者干脆卡在某个很高的值不再动弹。这正是深度神经网络训练中最核心、最棘手的问题之一。

这一课，我们将跟随Karpathy的步伐，像一个工程师调试火箭发动机那样，逐层打开深度网络的"引擎盖"，观察信号在网络中是如何流动的。我们会建立一整套诊断工具——激活值统计、梯度可视化、更新幅度监测——来量化网络的健康状况。更重要的是，我们会理解这些现象背后的数学原理，并引出改变深度学习进程的一项关键技术：**Batch Normalization**（批归一化）。

---

## 4.1 深度网络的训练困境

### 4.1.1 为什么加深网络会出问题：前向激活值衰减/爆炸，反向梯度消失/爆炸

想象一条水管。水流经过每一段管子，如果每段管子的直径恰好合适，水可以平稳地流到出口。但如果某段管子突然变窄很多（饱和），水流就会被截断；如果某段管子突然变宽很多（大权重），水流就会喷射而出。深度神经网络中的信号传播，正面临着这样的困境。

**前向传播中的激活值爆炸与衰减**。考虑一个最简单的深层网络，每一层只有线性变换，没有激活函数和偏置：

$$h^{(l)} = W^{(l)} h^{(l-1)}$$

假设输入 $h^{(0)}$ 的每个元素方差为1，$W^{(l)}$ 的每个元素独立采样自均值为0、方差为 $\sigma_w^2$ 的高斯分布。那么 $h^{(l)}$ 的第 $i$ 个元素是 $\sum_j W^{(l)}_{ij} h^{(l-1)}_j$，其方差为：

$$\text{Var}(h^{(l)}_i) = \sum_j \text{Var}(W^{(l)}_{ij}) \cdot \text{Var}(h^{(l-1)}_j) = n_{in} \cdot \sigma_w^2 \cdot \text{Var}(h^{(l-1)}_j)$$

这里 $n_{in}$ 是输入维度，也就是 fan_in。如果我们经过 $L$ 层，方差就会变成：

$$\text{Var}(h^{(L)}) = (n_{in} \cdot \sigma_w^2)^L \cdot \text{Var}(h^{(0)})$$

注意这个指数 $L$。如果 $n_{in} \cdot \sigma_w^2 > 1$，方差会**指数爆炸**；如果 $n_{in} \cdot \sigma_w^2 < 1$，方差会**指数衰减到零**。无论哪种情况，深度网络的输出都会变得毫无信息量。

激活函数的引入让情况更加复杂。以 tanh 为例：当输入较小时，tanh 近似线性，行为尚可预测；但当输入较大时，tanh 的输出被压缩到接近 +1 或 -1，进入**饱和区（saturation）**。饱和不仅限制了前向信号的动态范围，更严重的是——它杀死了反向传播的梯度。

**反向传播中的梯度消失与爆炸**。反向传播通过链式法则计算梯度。对于 tanh 激活，我们有：

$$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$

当 tanh 的输出接近 ±1 时，这个导数趋近于0。如果网络中很多神经元都处于饱和状态，反向传播时梯度会一层一层地被这些小数相乘，最终深层的梯度几乎为零——这就是**梯度消失（Vanishing Gradients）**。深层参数几乎不更新，网络"冻住了"。

反之，如果权重初始化过大，权重矩阵本身的奇异值可能大于1。反向传播时梯度向量反复乘以这些大奇异值的矩阵，导致梯度逐层放大——这就是**梯度爆炸（Exploding Gradients）**。loss 可能在某次更新后突然变成 NaN。

### 4.1.2 可视化工具：统计每层激活值的均值和标准差，绘制分布直方图

要诊断这些问题，我们需要看见网络内部发生了什么。Karpathy 在这一课中引入了一套简单但极其有效的可视化方法。

首先是**激活值分布直方图**。在每次前向传播后，我们收集每一层经过激活函数后的输出，统计它们的均值、标准差，以及有多少比例的神经元进入了饱和区。

```python
# 可视化各层激活值直方图
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out  # 获取tanh层的输出
        # 打印统计信息：均值、标准差、饱和比例
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % 
              (i, layer.__class__.__name__, t.mean(), t.std(), 
               (t.abs() > 0.97).float().mean()*100))
        # 绘制分布直方图
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')
```

这段代码遍历网络中的每一层，找到所有的 Tanh 激活层，获取它们的输出张量 `layer.out`。`t.abs() > 0.97` 这一行定义了饱和的判断标准——当激活值的绝对值超过0.97时，我们认为它进入了 tanh 的饱和边缘区域（因为 tanh(±2) ≈ ±0.96，已经非常接近极限）。饱和比例的百分比直接告诉我们这层有多少神经元"死掉了"。

### 4.1.3 诊断指标：追踪训练过程中各层统计量的变化

只看初始化时的激活分布是不够的。在训练过程中，随着权重的更新，激活分布会发生漂移。一个健康的网络应该在训练过程中保持各层激活分布的相对稳定——均值接近零，标准差维持在一个合理的范围内。

我们可以追踪以下几个关键诊断指标：

| 指标 | 正常范围 | 异常时的含义 |
|------|---------|------------|
| 激活值均值 | 接近 0 | 偏离0可能意味着偏置不当或内部协变量偏移 |
| 激活值标准差 | ~0.5-1.0 | 太小表示信号衰减，太大表示即将饱和 |
| 饱和比例 | < 10% | > 30% 意味着严重梯度消失 |
| 梯度均值 | ~0 | 大幅偏离0可能表示梯度爆炸方向偏移 |
| 梯度标准差 | 各层量级相近 | 前层比后层小很多=梯度消失，大很多=梯度爆炸 |
| 更新幅度比 (update:data) | ~1e-3 | 过大=学习率太大，过小=学习率太小或梯度消失 |

这些指标构成了我们诊断网络健康状况的"血常规检查"。在后续小节中，我们将通过具体的实验来观察不同初始化条件下这些指标的变化。

---

## 4.2 激活值分析

### 4.2.1 初始化时的激活分布：Kaiming初始化(2015)的理论与推导

让我们回到前向传播的方差分析。对于 tanh 激活函数，我们希望每一层输出的方差与输入的方差大致相同。这样可以避免信号在深层网络中指数级增长或衰减。

假设权重 $W$ 的每个元素独立同分布，均值为0，方差为 $\sigma_W^2$。输入 $x$ 的每个元素方差为 $\sigma_x^2$。那么线性变换 $y = Wx$ 的输出方差为：

$$\text{Var}(y_i) = \sum_{j=1}^{n_{in}} \text{Var}(W_{ij} x_j) = n_{in} \cdot \sigma_W^2 \cdot \sigma_x^2$$

为了让 $\text{Var}(y_i) = \sigma_x^2$，我们需要：

$$\sigma_W^2 = \frac{1}{n_{in}}$$

这就是**Xavier初始化**的核心结论。但对于 tanh 这类非线性激活函数，事情还有微妙之处。tanh 函数在0附近近似恒等映射，但当输入变大时会把值压缩。He et al. (2015) 的论文 "Delving Deep into Rectifiers" 中详细分析了这个问题，提出了**Kaiming初始化**[^1^]。

对于 tanh 激活，Kaiming 推荐的标准差为：

$$\sigma_W = \frac{1}{\sqrt{n_{in}}}$$

更进一步，考虑到 tanh 在0附近的斜率约为1，但随着输入范围的变化，其"有效增益"会下降。PyTorch 中使用的 `calculate_gain` 函数为 tanh 给出了增益因子 $\text{gain} = 5/3$，因此更精确的初始化标准差为：

$$\sigma_W = \frac{5/3}{\sqrt{n_{in}}}$$

不过 Karpathy 在课程中采用了更简化的版本 $\sigma_W = 1/\sqrt{n_{in}}$，这在实践中已经效果很好。

在代码中实现非常简单——我们在 `Linear` 层的 `__init__` 中将权重除以 `fan_in ** 0.5`：

```python
class Linear:
    """全连接层，使用Kaiming初始化"""
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming init: weight / sqrt(fan_in) 保持方差稳定
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

关键在第三行：`torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5`。如果不做这个除法，当 `fan_in = 30 * 10 = 300`（30个embedding维度拼接）时，标准差会从1膨胀到 $\sqrt{300} \approx 17$，经过几层tanh后几乎所有神经元都会立即饱和。

除了 `Linear` 和 `BatchNorm1d`，我们还需要一个简单的 `Tanh` 激活层。它本身没有可学习的参数，但保留输出以供后续检查激活分布：

```python
class Tanh:
    """Tanh激活函数层"""
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []  # Tanh层没有可学习参数
```

### 4.2.2 训练过程中的激活漂移：tanh饱和导致梯度消失

即使初始化完美，训练过程中激活分布仍可能漂移。想象一个场景：某一层的权重在训练过程中逐渐增大，导致送入 tanh 的 pre-activation 变大，越来越多的神经元进入饱和区。一旦进入饱和，梯度就会消失，这些神经元不再更新，形成了所谓的**"死神经元"**。

我们可以通过实验来观察这个现象。先构建一个深层网络（比如6层），用Kaiming初始化，训练几步后查看各层的饱和比例：

```python
# 构建深层MLP: 6层，每层Linear + BatchNorm + Tanh
n_embd = 10      # 字符嵌入向量的维度
n_hidden = 100   # MLP隐藏层的神经元数量

layers = [
    Linear(n_embd * block_size, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]
```

在训练初期，如果你使用Kaiming初始化，各层tanh输出的标准差应该在0.6到0.7之间（因为tanh把输入压缩），饱和比例应该很低（<5%）。但如果初始化过大，你会看到深层（前面几层）的饱和比例可能飙升到90%以上——几乎所有神经元都死掉了。

有趣的是，这种饱和往往呈现出一种**逐层递减**的模式：靠近输出的层因为接收到的梯度信号相对较强，可能还能维持部分活性；而靠近输入的层则因为梯度消失，权重几乎不更新，其激活分布完全由初始化和前面层传递过来的微弱信号决定。这就是为什么在深层网络中，梯度消失比梯度爆炸更难察觉——它表现为一种"安静的死亡"，而不是醒目的NaN。

### 4.2.3 权重矩阵的谱分析：奇异值分布揭示信息流动

除了激活值的分布，我们还可以从权重矩阵本身的性质来分析信息流动。对权重矩阵 $W$ 做**奇异值分解（SVD）**：$W = U \Sigma V^T$，其中 $\Sigma$ 是对角矩阵，对角线上的奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$ 描述了矩阵在不同方向上的缩放能力。

如果奇异值分布很集中（大部分奇异值大小相近），权重矩阵像一个均匀放大的透镜，信息可以均匀地通过。如果奇异值分布很分散（几个很大，其余很小），权重矩阵在某些方向上过度放大，在其他方向上则压缩——这会导致信息的扭曲和丢失。

在训练良好的网络中，权重矩阵的奇异值分布通常比较均匀。如果观察到奇异值分布严重不均匀（比如条件数 $\sigma_{max} / \sigma_{min}$ 非常大），可能意味着该层的训练出了问题。

---

## 4.3 梯度分析

如果说激活值分析是检查前向传播的"体检报告"，那么梯度分析就是检查反向传播的"心电图"。一个训练健康的网络，其梯度应该在各层之间保持合理的量级，既不会消失也不会爆炸。

### 4.3.1 反向传播中的梯度统计：每层梯度的大小和分布

为了观察梯度是如何在网络中流动的，我们需要在反向传播后收集每一层激活值上的梯度。PyTorch 默认不保留中间张量的梯度（为了节省内存），所以我们需要显式调用 `retain_grad()`：

```python
# backward pass
for layer in layers:
    layer.out.retain_grad()  # 保留中间激活值的梯度（用于调试可视化）
for p in parameters:
    p.grad = None
loss.backward()
```

`retain_grad()` 这行至关重要——它告诉 PyTorch 保留非叶子节点的梯度，这样我们就可以在反向传播后检查 `layer.out.grad`。

接下来可视化各层的梯度分布：

```python
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad  # 获取激活值上的梯度
        print('layer %d (%10s): mean %+f, std %e' % 
              (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('gradient distribution')
```

这段代码与激活值可视化的结构几乎一致，区别在于我们绘制的是 `layer.out.grad` 而非 `layer.out`。一个健康的网络中，各层梯度的标准差（`std %e` 打印的值）应该**大致处于同一数量级**。如果你发现从输出层到输入层，梯度标准差越来越小（比如第5层是 1e-3，第1层是 1e-7），这就是典型的**梯度消失**。反之，如果越来越大的则是梯度爆炸。

### 4.3.2 梯度与权重更新比：判断学习率是否合适的实用指标

仅仅观察梯度本身的大小还不够。一个更具实践意义的指标是**梯度与权重的比值**（grad:data ratio），也就是：

$$\text{ratio} = \frac{|\text{gradient}|_{std}}{|W|_{std}}$$

这个比值告诉我们：每次参数更新时，我们改变的幅度占参数本身幅度的多少。如果比值太大（比如 > 0.1），意味着我们在剧烈地改变参数，可能跳过最优解；如果太小（比如 < 1e-5），意味着参数几乎不动，学习太慢。

Karpathy 在课程中发现，一个合理的经验值是**update:data ratio 约为 1e-3**。这意味着每次更新只改变参数幅度的千分之一——足够小以保证稳定性，又足够大以推动优化前进。为什么是 1e-3？直觉上，如果一个参数的值大约是1，每次更新0.001，那么大约需要1000步才能让参数发生显著变化（比如从1变到0或2），这与训练迭代的总次数（通常是数万到数百万步）相匹配。

我们可以追踪训练过程中的这个比值：

```python
# 在训练循环中追踪 update:data ratio
with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() 
               for p in parameters])
```

这里 `(lr * p.grad).std()` 计算的是实际更新量（学习率乘以梯度）的标准差，`p.data.std()` 是参数值的标准差。我们取对数后绘制：

```python
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:  # 只绘制权重矩阵（忽略bias等一维参数）
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k')  # 1e-3 参考线 (log10(1e-3) = -3)
plt.legend(legends)
plt.title('update:data ratio over training (log scale)')
```

图中的黑线代表 $\log_{10}(10^{-3}) = -3$。理想情况下，各条曲线应该在这条黑线附近波动。如果所有曲线都远高于 -3，说明学习率太大；如果都远低于 -3，说明学习率太小或梯度已经消失。

### 4.3.3 可视化梯度直方图：识别梯度消失或爆炸

让我们将前面的分析综合起来，看看在实际训练中可能出现的几种典型情况。

**情况一：初始化过大**。权重标准差设为1（不做Kaiming缩放），前几步训练后你会发现：
- 激活值直方图：深层tanh的输出几乎全部集中在 ±1，饱和比例接近 100%
- 梯度直方图：梯度分布极其尖锐（标准差极小），因为 tanh 的导数几乎为零
- 结果：loss 完全不下降，网络陷入"僵尸状态"

**情况二：初始化过小**。权重标准差设为 0.01（过度缩放）：
- 激活值直方图：所有层的输出都集中在0附近一个非常窄的区间内
- 梯度直方图：虽然梯度不为零，但从输出层到输入层逐级衰减
- 结果：loss 下降极其缓慢，深层参数几乎不更新

**情况三：Kaiming初始化 + 合理的深度**。这是我们追求的目标：
- 激活值直方图：各层tanh输出分布合理，类似一个拉宽的"M"形，饱和比例低
- 梯度直方图：各层梯度标准差处于同一数量级
- 结果：loss 稳步下降，训练高效且稳定

这些可视化方法虽然简单，却是深度学习工程师日常调试的利器。当你遇到一个不收敛的深度网络时，打开这些直方图，问题往往一目了然。

---

## 4.4 Batch Normalization

### 4.4.1 BatchNorm的动机：internal covariate shift，每层输入分布不稳定

我们已经看到，即使使用Kaiming初始化，深度网络的训练仍然很脆弱。权重初始化的微小偏差、学习率的选择、甚至训练数据批次的不同排列，都可能导致训练失败。根本原因在于：**在训练过程中，每一层输入的分布会不断变化**。

考虑网络中的第3层。当第1层和第2层的权重在训练过程中更新时，第3层接收到的输入分布也在随之改变。第3层必须不断适应这种变化的输入分布——论文作者 Ioffe 和 Szegedy 将这种层与层之间的分布漂移称为**内部协变量偏移（Internal Covariate Shift）**[^2^]。

这个术语借用自机器学习领域更广泛的概念"协变量偏移（Covariate Shift）"，指的是训练时和测试时输入分布不一致。而这里的"内部"指的是网络内部的层与层之间的分布不一致。每一次前面层参数的更新，都会改变后续层面对的输入分布，这使得深层网络的训练像在一个不断晃动的平衡木上走路。

Batch Normalization 的核心思想极其简洁：**为什么不直接在每一层把输入分布"固定"下来呢？** 具体来说，对每一层的 pre-activation（激活函数之前的值），减去当前 batch 的均值，除以当前 batch 的标准差，使其变成均值为0、方差为1的标准正态分布。这样，无论前面层怎么变，后面层接收到的输入分布至少是稳定的。

### 4.4.2 BatchNorm的数学公式：减均值除以标准差，再学两个参数scale和shift

BatchNorm 的完整前向传播公式如下。给定一个 batch 的输入 $x \in \mathbb{R}^{B \times D}$（$B$ 是 batch size，$D$ 是特征维度）：

**第一步：计算 batch 统计量**

$$\mu_B = \frac{1}{B} \sum_{i=1}^{B} x_i \quad \text{(batch 均值)}$$

$$\sigma^2_B = \frac{1}{B} \sum_{i=1}^{B} (x_i - \mu_B)^2 \quad \text{(batch 方差)}$$

**第二步：标准化**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$$

其中 $\epsilon$ 是一个很小的常数（通常 $10^{-5}$），防止除以零。

**第三步：缩放和平移**

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

这里 $\gamma$（scale）和 $\beta$（shift）是可学习的参数。为什么要加这一步？因为强制标准化到均值为0、方差为1可能会限制网络的表达能力。比如对于 sigmoid/tanh 这类激活函数，如果输入总是标准正态分布，那只会用到激活函数中间很小的一段线性区域。通过学习 $\gamma$ 和 $\beta$，网络可以恢复任意均值和方差的分布，BatchNorm 只是提供了一种"稳定的基础"。

注意 $\gamma$ 和 $\beta$ 是对每个特征维度单独学习的，所以它们的维度与特征维度 $D$ 相同。

### 4.4.3 BatchNorm的实现：训练时使用当前batch统计，测试时使用全局统计

BatchNorm 在训练和推理时的行为是不同的，这是理解它的关键。

**训练时**：使用当前 batch 的均值和方差进行标准化。同时，维护**全局运行统计量（running statistics）**，通过指数移动平均更新：

$$\text{running\_mean} = (1 - \text{momentum}) \cdot \text{running\_mean} + \text{momentum} \cdot \mu_B$$

$$\text{running\_var} = (1 - \text{momentum}) \cdot \text{running\_var} + \text{momentum} \cdot \sigma^2_B$$

这里的 `momentum` 通常设为 0.1。注意这个更新是在 `torch.no_grad()` 上下文中进行的，不参与反向传播。

**推理/测试时**：不再使用 batch 统计（因为单个样本或小 batch 的统计不稳定），而是使用训练期间累积的 running statistics：

$$\hat{x} = \frac{x - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}}$$

$$y = \gamma \cdot \hat{x} + \beta$$

这种模式切换通过 `self.training` 标志来控制，与 PyTorch 的 `model.train()` / `model.eval()` 约定一致。

以下是完整的 BatchNorm1d 实现：

```python
class BatchNorm1d:
    """一维Batch Normalization层，从零实现"""
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True  # 训练/推理模式切换
        # 可学习参数 (通过反向传播训练)
        self.gamma = torch.ones(dim)   # scale
        self.beta = torch.zeros(dim)   # shift
        # buffers (通过running momentum update更新，不参与梯度)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            # 训练时：使用当前batch的统计量
            xmean = x.mean(0, keepdim=True)       # batch均值
            xvar = x.var(0, keepdim=True)         # batch方差（注意是0号维度）
        else:
            # 推理时：使用训练期间累积的全局统计量
            xmean = self.running_mean
            xvar = self.running_var
        
        # 标准化到单位方差
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # 更新running statistics（仅在训练时）
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
```

几个关键细节值得注意。`x.mean(0, keepdim=True)` 在0号维度（batch维度）上求均值，所以输出形状是 `[1, D]`。`keepdim=True` 保持了二维形状，便于后续广播减法。`running_mean` 和 `running_var` 被称为**buffer**而非参数——它们不通过梯度下降更新，而是通过前向传播中的动量更新来维护。`self.training` 标志控制了训练和推理的不同路径。

以下是完整的训练循环代码，包含了所有诊断工具：

```python
max_steps = 200000
batch_size = 32
lossi = []
ud = []  # update to data ratio

for i in range(max_steps):
    # minibatch构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    # forward pass
    emb = C[Xb]                           # 嵌入字符到向量
    x = emb.view(emb.shape[0], -1)       # 拼接向量
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)         # 损失函数
    
    # backward pass
    for layer in layers:
        layer.out.retain_grad()           # 保留中间激活值的梯度（用于调试可视化）
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 参数更新，带学习率衰减
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    # 追踪统计
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() 
                   for p in parameters])
```

### 4.4.4 BatchNorm为什么有效：平滑损失曲面、允许更大学习率、有轻微正则化效果

BatchNorm 为什么能让深度网络的训练变得如此容易？原始论文将其归功于"减少了内部协变量偏移"，但后续研究对此提出了更深入的解释。

**"How Does Batch Normalization Help Optimization?"**[^3^] 这篇论文指出，BatchNorm 有效的主要原因不是减少内部协变量偏移，而是它**平滑了损失函数的梯度曲面**。想象你在一个崎岖不平的山坡上往谷底走——BatchNorm 就像是把山坡"熨平"了，使得优化器可以更大胆地迈步而不至于跌下悬崖。

具体来说，BatchNorm 让损失函数对权重缩放的敏感度大大降低。在没有BN的网络中，如果某一层的权重突然增大两倍，下一层接收到的激活值也会增大两倍，loss 可能剧烈变化。有了BN之后，即使权重增大，BN层会把激活值标准化回相似的分布，loss 的变化被抑制了。这种**Lipschitz 连续性的提升**使得优化更加稳定。

此外，BatchNorm 还有两个实用的附带效果：

**允许使用更大的学习率**。在没有BN的网络中，学习率稍大就可能导致梯度爆炸或激活值饱和。BN通过将激活值限制在合理范围内，大大提高了训练对学习率的容忍度。

**轻微的正则化效果**。因为每个样本的标准化依赖于同 batch 中其他样本的统计量，这引入了一定的随机性——同一个样本在不同 batch 中会被标准化成略有不同的值。这种"噪声"起到了类似 Dropout 的正则化效果，有助于防止过拟合。不过，这种正则化效果比较轻微且不可控，不应作为主要的正则化手段。

值得注意的是，BatchNorm 改变了网络的表达能力。一个有趣的事实是：带有BN的网络在初始化时的输出不再依赖于权重的具体缩放——因为BN会把任何分布都标准化回去。这意味着BN网络对初始化不敏感，这正是我们想要的。

### 4.4.5 手写BatchNorm层：从scratch实现含反向传播的完整BatchNorm

上面的 `BatchNorm1d` 类依赖于 PyTorch 的 autograd 来自动计算梯度。但作为学习者，理解 BatchNorm 的反向传播对于深入掌握神经网络至关重要。让我们推导并手写完整的反向传播。

BatchNorm 的前向传播可以分解为以下计算图：

$$x \rightarrow \mu_B = \text{mean}(x) \rightarrow x_{\text{c}} = x - \mu_B \rightarrow \sigma^2_B = \text{mean}(x_c^2) \rightarrow \hat{x} = x_c / \sqrt{\sigma^2_B + \epsilon} \rightarrow y = \gamma \cdot \hat{x} + \beta$$

反向传播时，我们需要计算每个中间变量对输入 $x$ 的梯度。设上游传来的梯度为 $dy$（即 loss 对 $y$ 的梯度），我们需要依次求出：

**$dy \rightarrow d\hat{x}$**: 由于 $y = \gamma \cdot \hat{x} + \beta$：

$$d\hat{x} = dy \cdot \gamma$$

**$d\hat{x} \rightarrow dx_c$**: 由于 $\hat{x} = x_c / \sqrt{\sigma^2_B + \epsilon}$：

$$dx_c^{(1)} = d\hat{x} / \sqrt{\sigma^2_B + \epsilon}$$

**$d\hat{x} \rightarrow d\sigma^2_B$**: 同理：

$$d\sigma^2_B = \text{mean}\left(d\hat{x} \cdot x_c \cdot \left(-\frac{1}{2}\right) \cdot (\sigma^2_B + \epsilon)^{-3/2}\right)$$

**$d\sigma^2_B \rightarrow dx_c^{(2)}$**: 由于 $\sigma^2_B = \text{mean}(x_c^2)$：

$$dx_c^{(2)} = d\sigma^2_B \cdot 2 \cdot x_c / B$$

**$dx_c \rightarrow d\mu_B$**: 由于 $x_c = x - \mu_B$：

$$d\mu_B = -\text{mean}(dx_c^{total})$$

**$dx_c^{total} \rightarrow dx$**: 合并所有经过 $x_c$ 的路径：

$$dx = dx_c^{total} + d\mu_B / B$$

最终的完整实现如下：

```python
class BatchNorm1dFull:
    """BatchNorm1d 的完整手动实现（含反向传播）"""
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # 可学习参数
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        # 前向传播
        xstd = torch.sqrt(xvar + self.eps)
        xhat = (x - xmean) / xstd
        out = self.gamma * xhat + self.beta
        
        # 更新 running stats
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
            # 保存中间变量用于反向传播
            self.cache = (x, xmean, xvar, xstd, xhat, self.gamma)
        
        return out
    
    def backward(self, dout):
        """手动实现BatchNorm的反向传播
        dout: 上游梯度，形状 [B, D]
        返回: dx, 形状 [B, D]
        """
        x, xmean, xvar, xstd, xhat, gamma = self.cache
        B = x.shape[0]
        
        # dgamma 和 dbeta
        self.gamma.grad = (dout * xhat).sum(0)
        self.beta.grad = dout.sum(0)
        
        # dxhat
        dxhat = dout * gamma
        
        # dx (主路径 + 经过mean和var的路径)
        dx = dxhat / xstd
        dvar = (dxhat * (x - xmean) * (-0.5) * (xvar + self.eps)**(-1.5)).sum(0, keepdim=True)
        dmean = (-dxhat / xstd).sum(0, keepdim=True) + dvar * (-2) * (x - xmean).mean(0, keepdim=True)
        
        dx += dvar * 2 * (x - xmean) / B + dmean / B
        
        return dx
```

这个手动实现的 `backward` 方法展示了 BatchNorm 反向传播的核心逻辑。虽然 PyTorch 的 autograd 可以自动完成这些计算，但手写一遍能让我们真正理解梯度是如何流经标准化操作的。

### 4.4.6 Inference时的BatchNorm折叠：将BN参数融合进相邻线性层

BatchNorm 的一个美妙性质是：在推理时，它其实可以被"折叠"到相邻的线性层中，完全消除额外的计算开销。

考虑一个 Linear 层后接 BatchNorm 层的组合：

$$h = W \cdot x + b \quad \text{(Linear)}$$
$$y = \gamma \cdot \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \quad \text{(BatchNorm)}$$

将 Linear 的输出代入 BatchNorm：

$$y = \gamma \cdot \frac{W \cdot x + b - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
$$= \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W \cdot x + \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b - \mu) + \beta$$

令：

$$W' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W$$
$$b' = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b - \mu) + \beta$$

那么：

$$y = W' \cdot x + b'$$

这就是纯粹的线性变换！推理时我们只需要用更新后的 $W'$ 和 $b'$ 做一次矩阵乘法即可，不需要计算均值、方差、标准化等操作。

```python
# BatchNorm折叠到Linear层
def fold_bn_into_linear(linear, bn):
    """将BatchNorm的参数折叠到相邻的Linear层中
    
    Args:
        linear: Linear层
        bn: BatchNorm1d层
    
    Returns:
        折叠后的新的weight和bias
    """
    std = torch.sqrt(bn.running_var + bn.eps)
    
    # 新权重 = gamma / std * W
    new_weight = bn.gamma / std * linear.weight
    
    # 新偏置 = gamma / std * (bias - running_mean) + beta
    if linear.bias is not None:
        new_bias = bn.gamma / std * (linear.bias - bn.running_mean) + bn.beta
    else:
        new_bias = bn.beta - bn.gamma * bn.running_mean / std
    
    return new_weight, new_bias
```

这个折叠技巧在模型部署中非常重要。比如一个 ResNet-50 网络中有53个 BatchNorm 层，全部折叠后推理速度可以显著提升。这也证明了 BatchNorm 本质上是一种"训练时技巧"——它的存在是为了让训练更稳定，但在一个训练好的网络中，它的作用完全可以被相邻层吸收。

---

## 4.5 课后练习与资源

### 4.5.1 练习题

**E01: 从零初始化实验**

将所有权重和 bias 初始化为零，训练网络。回答以下问题：

- 网络能否训练？loss 是否下降？下降到多少？
- 观察各层的激活值分布和梯度分布，哪些层在被训练，哪些层没有？
- 分析为什么网络只能"部分"训练——提示：考虑对称性破坏（symmetry breaking）的概念
- 这个实验说明了什么？为什么神经网络初始化时不能全为零？

**E02: BatchNorm推理折叠验证**

实现并验证 BatchNorm 的推理折叠：

- 训练一个带 BatchNorm 的3层MLP（embedding → Linear → BN → Tanh → Linear → BN → Tanh → Linear → BN）
- 训练完成后，将每个 BatchNorm 层的参数按公式折叠到前一个 Linear 层中
- 构建一个"折叠后"的网络（只有 Linear + Tanh，没有 BN）
- 用相同的数据分别通过原网络（eval模式）和折叠后网络，验证两者输出的 loss 完全一致
- 这个实验证明了什么？为什么说 BatchNorm 只是训练稳定器，不是表达能力的增强？

### 4.5.2 推荐学习资源

**论文**

| 资源 | 链接 | 说明 |
|------|------|------|
| **BatchNorm原始论文** | https://arxiv.org/abs/1502.03167 | Ioffe & Szegedy 2015，"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"，这是必读的里程碑论文[^2^] |
| **Kaiming初始化论文** | https://arxiv.org/abs/1502.01852 | He et al. 2015，"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"，详细推导了ReLU/tanh等激活函数下的初始化理论[^1^] |
| **BatchNorm问题分析** | https://arxiv.org/abs/2105.07576 | Santurkar et al. 2018，"How Does Batch Normalization Help Optimization?"，挑战了原始论文的解释，指出BN的主要作用是平滑损失曲面而非减少ICS[^3^] |

**视频与代码**

| 资源 | 链接 |
|------|------|
| **Karpathy第4课视频** | https://www.youtube.com/watch?v=P6sfmUTpUmc |
| **GitHub Notebook** | https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb |
| **makemore项目** | https://github.com/karpathy/makemore |
| **系列播放列表** | https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ |




---

## 5. 第5课：makemore Part 4 — 成为反向传播忍者

在前面的课程中，我们已经搭建了一个带有BatchNorm的两层MLP，并且通过PyTorch的autograd让网络学会了生成人名。一切看起来都很美好：定义好前向传播、调用`loss.backward()`、参数就会自动获得正确的梯度。但这节课，Karpathy要我们做一件看起来近乎"疯狂"的事情：**彻底抛弃`loss.backward()`，自己手写每一层、每一个操作的反向传播**。这个过程虽然繁琐，却是整个系列中最具启发性的一课——当你亲手推导出BatchNorm的梯度公式，或发现Cross Entropy Loss的梯度原来如此优雅时，你对神经网络的理解将跃升到一个全新的层次。

---

### 5.1 手动反向传播的意义

#### 5.1.1 为什么不用autograd

现代深度学习框架（PyTorch、TensorFlow、JAX）的一个核心优势就是自动微分（automatic differentiation，简称autograd）。你只需定义前向传播的计算图，框架就能自动沿着计算图反向传播，计算出每个参数相对于损失的梯度。这极大地降低了深度学习的研究门槛——你不需要是微积分专家，也能训练出效果不错的神经网络。

但autograd是一把双刃剑。它的便利性带来了一个隐蔽的问题：**它让用户对"梯度是如何在计算图中流动的"这件事失去了直觉**。当你的网络训练异常时（损失不下降、出现NaN、梯度爆炸或消失），如果你不知道梯度在每一层长什么样、应该长什么样，你就只能盲目地调参，像在黑箱外敲敲打打。而一个真正理解反向传播的人，则能够通过`print`每一层的梯度统计信息，迅速定位问题所在。

Karpathy在本课开头分享了一个有趣的历史细节：反向传播算法（Backpropagation）虽然在1986年由Rumelhart、Hinton和Williams推广开来[^1^]，但它的核心思想——链式法则（Chain Rule）——可以追溯到17世纪的莱布尼茨。直到今天，仍有一些研究者试图"重新发明"反向传播的替代品（如前向梯度计算），但这些尝试在计算效率上往往难以与经典的反向传播相抗衡。理解反向传播不仅是调试工具，更是理解深度学习如何工作的核心窗口。

#### 5.1.2 手动backprop的能力边界

手动实现反向传播能带来什么实际好处？让我们看几个具体的场景。

**场景一：调试网络**。假设你设计了一个新的激活函数或损失函数，PyTorch中并没有现成的实现。你必须自己写它的前向和反向传播。如果你不熟悉如何为一个标量函数推导梯度，这个功能就无法实现。本课的Exercise 2中，我们将看到Cross Entropy Loss（softmax + negative log-likelihood的组合）的梯度推导出一个令人惊讶的简洁形式——这种洞察力只有亲手推导才能获得。

**场景二：修改梯度流**。许多重要的深度学习技术涉及对梯度流的直接操作。梯度裁剪（gradient clipping）在RNN训练中防止梯度爆炸；meta-learning中的MAML算法需要计算梯度的梯度；一些最新的优化器研究直接修改参数的更新方向。这些操作都要求你对"梯度是什么、在哪里、如何传递"有清晰的认知。

**场景三：理解架构设计的权衡**。为什么BatchNorm能让网络训练得更快更稳定？为什么ResNet中的跳跃连接（skip connection）如此有效？这些问题的答案都藏在梯度流中。BatchNorm的本质是让每一层的输入分布保持零均值和单位方差，这防止了深层网络中梯度信号的衰减；ResNet的跳跃连接则创建了一条梯度直接回传的"高速公路"。当你亲手推导过这些层的反向传播公式后，这些架构设计的动机就会显得无比自然。

#### 5.1.3 学习目标

本课的核心目标是：**能手推网络中每个操作的局部雅可比矩阵（local Jacobian）**。具体来说，当给定一个上游梯度`dL/dy`（损失对某个操作输出$y$的梯度）时，你需要能够计算出`dL/dx`（损失对该操作输入$x$的梯度）以及`dL/dW`、`dL/db`（损失对该操作内部参数的梯度）。

这个过程始终遵循链式法则：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

其中$\partial y / \partial x$是操作本身的局部导数（有时是一个矩阵，即雅可比矩阵），而$\partial L / \partial y$是从损失端反向传递过来的上游梯度。我们的任务就是为计算图中的每一个节点，正确计算出这个局部导数，并与上游梯度相乘，然后将结果继续向前传递。

---

### 5.2 逐层手动反向传播

在开始推导之前，我们需要引入一个关键的工具函数，它将贯穿整节课的验证过程。

#### 5.2.0 工具函数：`cmp()`梯度对比

为了确保手动计算的梯度与PyTorch autograd计算的梯度完全一致，我们定义一个`cmp()`函数来对比两者：

```python
def cmp(s, dt, t):
    """
    比较手动计算的梯度 dt 与 PyTorch autograd 的梯度 t.grad
    s: 变量名称字符串
    dt: 手动计算的梯度
    t: PyTorch张量（其 .grad 属性包含 autograd 计算的梯度）
    """
    ex = torch.all(dt == t.grad).item()          # 完全相等？
    app = torch.allclose(dt, t.grad)             # 近似相等？（考虑浮点误差）
    maxdiff = (dt - t.grad).abs().max().item()   # 最大差异
    print(f'{s:15s} | exact: {str(ex):5s} | '
          f'approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

这个函数打印三列信息：两个梯度是否完全相等（通常因浮点运算顺序差异而为`False`）、是否在数值容差范围内近似相等（应为`True`），以及两者之间的最大差异（应是一个非常小的数，如$10^{-8}$量级）。每当完成一层的手动反向传播后，我们都会调用`cmp()`来验证，确保没有引入任何错误。

#### 5.2.1 分块化的前向传播

手动反向传播的第一步，是把前向传播拆解成尽可能小的"块"（chunk）。每一块只执行一个原子操作（如加法、乘法、矩阵乘法、逐元素函数等），并将中间结果保存在一个命名变量中。这样做的原因是：每一块对应一个局部梯度计算，块越小，局部梯度越简单，推导越不容易出错。

让我们先看看前向传播的完整代码，注意每一步都保留了中间变量：

```python
# 网络参数（与上一课相同的两层MLP + BatchNorm）
n_embd = 10    # 字符嵌入维度
n_hidden = 64  # 隐藏层神经元数量
n = batch_size # batch中的样本数

# ---- 前向传播（分块化） ----

# 1) Embedding层
emb = C[Xb]                              # (n, block_size, n_embd)
embcat = emb.view(emb.shape[0], -1)      # (n, block_size * n_embd)

# 2) 第1层线性变换
hprebn = embcat @ W1 + b1                # (n, n_hidden)，预BatchNorm的隐藏层激活

# 3) BatchNorm层（逐元素展开）
bnmeani = 1/n * hprebn.sum(0, keepdim=True)         # (1, n_hidden)，batch均值
bndiff = hprebn - bnmeani                            # (n, n_hidden)，去均值
bndiff2 = bndiff ** 2                                # (n, n_hidden)，平方
bnvar = 1/(n-1) * (bndiff2).sum(0, keepdim=True)     # (1, n_hidden)，样本方差
bnvar_inv = (bnvar + 1e-5) ** -0.5                   # (1, n_hidden)，标准差的倒数
bnraw = bndiff * bnvar_inv                           # (n, n_hidden)，归一化后的值
hpreact = bngain * bnraw + bnbias                    # (n, n_hidden)，缩放+平移

# 4) Tanh激活函数
h = torch.tanh(hpreact)                  # (n, n_hidden)

# 5) 第2层线性变换
logits = h @ W2 + b2                     # (n, vocab_size)

# 6) Cross Entropy Loss（逐元素展开）
logit_maxes = logits.max(1, keepdim=True).values     # (n, 1)，数值稳定性：取每行最大值
norm_logits = logits - logit_maxes                   # (n, vocab_size)，减去最大值
counts = norm_logits.exp()                           # (n, vocab_size)，指数化
counts_sum = counts.sum(1, keepdims=True)            # (n, 1)，求和
counts_sum_inv = counts_sum ** -1                     # (n, 1)，求倒数
probs = counts * counts_sum_inv                      # (n, vocab_size)，softmax概率
logprobs = probs.log()                               # (n, vocab_size)，取对数
loss = -logprobs[range(n), Yb].mean()                # 标量，取目标位置的负对数似然均值
```

注意BatchNorm中方差的分母用的是`n-1`而非`n`，这被称为Bessel's Correction（贝塞尔校正），是统计学中对样本方差的无偏估计。原始的BatchNorm论文[^2^]中实际使用的是`n`，但许多现代实现（包括PyTorch的默认行为）采用`n-1`。了解这一点很重要，因为分母的选择会影响梯度的具体形式。

在调用`loss.backward()`之前，我们需要对所有中间变量调用`retain_grad()`，这样PyTorch才会保留它们的梯度（默认情况下只有叶子节点的梯度会被保存）：

```python
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
loss.backward()
```

现在，`logprobs.grad`、`probs.grad`等所有中间变量的梯度都已经由PyTorch计算好了。我们的任务是从`loss`开始，一步步手动推导出每个变量的梯度，然后用`cmp()`来验证。

#### 5.2.2 Cross Entropy Loss的梯度推导

让我们从最简单的地方开始——损失函数本身。前向传播的最后一步是：

```python
loss = -logprobs[range(n), Yb].mean()
```

这里`logprobs`的形状是`(n, vocab_size)`，其中`n`是batch size。`logprobs[range(n), Yb]`取出每个样本对应正确标签位置的对数概率，然后取均值并取负号。展开来看：

$$\text{loss} = -\frac{1}{n} \sum_{i=0}^{n-1} \text{logprobs}[i, Y_b[i]]$$

这意味着损失直接依赖于`logprobs`中`n`个特定位置（即每个样本的正确标签位置）的值。对于`logprobs`中的任意一个元素`logprobs[i, j]`，如果`j == Yb[i]`（是该样本的正确标签），则损失对它的导数是$-1/n$；否则是$0$。这就是`dlogprobs`的起源：

```python
dlogprobs = torch.zeros_like(logprobs)   # 先创建全零矩阵
dlogprobs[range(n), Yb] = -1.0 / n       # 在正确标签位置填入 -1/n
```

接下来是`logprobs`到`probs`的反向传播。前向代码是`logprobs = probs.log()`，即逐元素取自然对数。局部导数是`d(logprobs)/d(probs) = 1/probs`。根据链式法则：

```python
dprobs = (1.0 / probs) * dlogprobs
```

这里`1.0 / probs`是局部导数，`dlogprobs`是上游梯度，两者逐元素相乘得到`dprobs`。

`probs`的生成涉及两个步骤的前向传播：`counts_sum_inv = counts_sum**-1`和`probs = counts * counts_sum_inv`。这是一个乘法操作，其中`counts`的形状是`(n, vocab_size)`，`counts_sum_inv`的形状是`(n, 1)`（通过广播机制扩展到`(n, vocab_size)`）。这里要格外小心：`probs`同时依赖于`counts`和`counts_sum_inv`，因此每个变量都会收到一条梯度路径。

对`counts_sum_inv`求导：把`counts`看作常数，$\partial \text{probs} / \partial \text{counts\_sum\_inv} = \text{counts}$。由于`counts_sum_inv`在每一步中被广播到所有`vocab_size`个位置，我们需要把所有位置的梯度加起来（这就是`sum(1, keepdim=True)`的作用）：

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
```

对`counts`的求导则有两个来源。第一个来源是直接从`probs = counts * counts_sum_inv`来的：把`counts_sum_inv`看作常数，局部导数是`counts_sum_inv`。第二个来源是`counts`还参与了`counts_sum = counts.sum()`，而`counts_sum`又影响了`counts_sum_inv`。我们先处理第一个来源：

```python
dcounts = counts_sum_inv * dprobs   # 第一条路径
```

然后处理`counts_sum_inv`到`counts_sum`的反向传播。前向代码`counts_sum_inv = counts_sum**-1`，局部导数是$-\text{counts\_sum}^{-2}$：

```python
dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
```

`counts_sum`是`counts`沿`vocab_size`维度的求和，所以它的梯度会被广播回`counts`的每个元素：

```python
dcounts += torch.ones_like(counts) * dcounts_sum   # 第二条路径累加到dcounts
```

现在`dcounts`汇聚了两条路径的梯度。接下来是`counts = norm_logits.exp()`。指数函数的局部导数是它自身：`d(counts)/d(norm_logits) = counts`。因此：

```python
dnorm_logits = counts * dcounts
```

`norm_logits = logits - logit_maxes`是一个减法操作。`logit_maxes`是为了数值稳定性而引入的（防止指数爆炸），它每行取`logits`的最大值。`logits`有两条梯度路径：一条直接从`norm_logits = logits - logit_maxes`来，另一条通过`logit_maxes`间接来。

第一条路径很简单：减法操作对`logits`的局部导数是$1$，所以`dnorm_logits`直接复制给`dlogits`的一个副本：

```python
dlogits = dnorm_logits.clone()
```

第二条路径：`logit_maxes`对`norm_logits`的影响是每行减去最大值，所以`dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)`（负号来自减法，sum是因为`logit_maxes`每行的单个值被广播到了所有`vocab_size`列）。然后`logit_maxes`是`logits.max()`的结果，它的梯度需要被路由回`logits`中最大值所在的那些位置。我们用`F.one_hot`来创建一个one-hot掩码，将梯度精确地放置在每行最大值的位置上：

```python
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
```

至此，我们已经完成了从损失函数一路反向传播到`logits`的全部过程。虽然步骤看起来很多，但每一步都只涉及链式法则的基本应用：局部导数乘以上游梯度。下面是Cross Entropy Loss部分完整的手动反向传播代码，附带验证：

```python
# ===== Cross Entropy Loss 的完整手动反向传播 =====

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n

dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs
dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = counts * dcounts
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

# 验证
cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
```

运行后，`cmp()`应该显示所有`approximate`列为`True`，`maxdiff`为$10^{-8}$量级或更小。这意味着我们手动推导的Cross Entropy Loss梯度与PyTorch autograd的结果完全一致。

#### 5.2.3 第2层Linear层的反向传播

接下来从`logits`继续反向传播，经过第2层线性层。前向传播是`logits = h @ W2 + b2`。这是矩阵乘法加偏置的经典线性层。

回忆矩阵乘法的梯度规则：如果$Y = XW$，则$dX = dY \cdot W^T$，$dW = X^T \cdot dY$。直觉上，$dX$的形状必须与$X$相同，而$dY \cdot W^T$的形状正好是`(n, out) @ (out, in) = (n, in)`，与$X$的`(n, in)`匹配。同理$dW = X^T \cdot dY$的形状是`(in, n) @ (n, out) = (in, out)`，与$W$的形状匹配。

对于偏置$b2$，前向传播中它被广播到每一行，反向传播时梯度需要沿batch维度累加：

```python
dh = dlogits @ W2.T           # (n, vocab_size) @ (vocab_size, n_hidden) = (n, n_hidden)
dW2 = h.T @ dlogits           # (n_hidden, n) @ (n, vocab_size) = (n_hidden, vocab_size)
db2 = dlogits.sum(0)          # (vocab_size,)，沿batch维度求和
```

验证：

```python
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
```

#### 5.2.4 Tanh激活的反向传播

通过线性层后，梯度来到了`h`，需要继续反向传播通过Tanh激活函数。前向代码是`h = torch.tanh(hpreact)`。Tanh函数的导数有一个优美的形式：

$$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$

这个公式可以通过Tanh的定义$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$直接求导得到。既然`h`就是`tanh(hpreact)`的值，我们可以直接用`h`来计算导数，无需重新计算Tanh：

```python
dhpreact = (1.0 - h ** 2) * dh
```

这里`1.0 - h ** 2`是局部导数，`dh`是上游梯度。Tanh的梯度有一个重要的性质：当`h`接近$\pm 1$时，`1 - h^2`接近$0$，梯度被"压缩"了。这就是Tanh（以及Sigmoid）容易出现梯度消失问题的原因——如果激活值饱和在两端，反向传播经过这一层时梯度几乎被消灭。

```python
cmp('hpreact', dhpreact, hpreact)
```

#### 5.2.5 BatchNorm的反向传播

BatchNorm是整节课中最复杂的部分。我们需要同时处理四条梯度路径：对`bngain`（gamma）的梯度、对`bnbias`（beta）的梯度、对`bnraw`的梯度，以及对`hprebn`的梯度。

前向代码：

```python
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1) * (bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
```

从`hpreact`往回推。`hpreact = bngain * bnraw + bnbias`中，`bngain`和`bnbias`是可学习的参数，`bnraw`是中间变量。

对`bngain`（gamma）求导：`bngain`与`bnraw`逐元素相乘，所以局部导数是`bnraw`。由于`bngain`被广播到batch中所有样本，梯度需要沿batch维度累加：

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
```

对`bnbias`（beta）求导：偏置被加到每个元素上，所以梯度沿batch和特征维度全部累加：

```python
dbnbias = dhpreact.sum(0, keepdim=True)
```

对`bnraw`求导：`bngain`是乘数，所以局部导数是`bngain`：

```python
dbnraw = bngain * dhpreact
```

现在是最复杂的部分：从`bnraw`反向传播到`bndiff`和`bnvar_inv`。`bnraw = bndiff * bnvar_inv`，两者都是乘法因子，所以各自获得一条梯度路径。

`bndiff`的梯度：`bnvar_inv`是局部导数：

```python
dbndiff = bnvar_inv * dbnraw
```

`bnvar_inv`的梯度：`bndiff`是局部导数，需要沿batch维度求和（因为`bnvar_inv`被广播到所有`n`行）：

```python
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

继续从`bnvar_inv`到`bnvar`。`bnvar_inv = (bnvar + 1e-5) ** -0.5`，这是一个幂函数，局部导数是$-0.5 \cdot (\text{bnvar} + 10^{-5})^{-1.5}$：

```python
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
```

从`bnvar`到`bndiff2`。`bnvar = 1/(n-1) * bndiff2.sum(0)`，所以`bndiff2`中每个元素的局部导数都是`1/(n-1)`：

```python
dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
```

注意`bndiff`现在有两条梯度路径了：一条从`bnraw = bndiff * bnvar_inv`来（已存入`dbndiff`），另一条从`bndiff2 = bndiff ** 2`来。第二条路径的局部导数是$2 \cdot \text{bndiff}$：

```python
dbndiff += (2 * bndiff) * dbndiff2   # 累加第二条路径
```

现在从`dbndiff`继续往`hprebn`和`bnmeani`传播。`bndiff = hprebn - bnmeani`，减法操作对`hprebn`的局部导数是$1$，对`bnmeani`的局部导数是$-1$：

```python
dhprebn = dbndiff.clone()            # hprebn的第一条路径
dbnmeani = (-dbndiff).sum(0)         # bnmeani的梯度（被广播，所以sum）
```

最后，`bnmeani = 1/n * hprebn.sum(0)`，`hprebn`中每个元素对`bnmeani`的贡献是`1/n`，所以`bnmeani`的梯度被均匀分摊回`hprebn`的每个元素：

```python
dhprebn += 1.0 / n * (torch.ones_like(hprebn) * dbnmeani)
```

现在`dhprebn`也汇聚了两条路径的梯度。BatchNorm反向传播的完整代码如下：

```python
# ===== BatchNorm 的完整手动反向传播 =====

dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = bngain * dhpreact

dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
dbndiff += (2 * bndiff) * dbndiff2

dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0)
dhprebn += 1.0 / n * (torch.ones_like(hprebn) * dbnmeani)

# 验证
cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw', dbnraw, bnraw)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiff2', dbndiff2, bndiff2)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('hprebn', dhprebn, hprebn)
```

#### 5.2.6 第1层Linear层的反向传播

通过BatchNorm后，梯度`dhprebn`需要继续反向传播经过第1层线性层。推导过程与第2层完全对称：

```python
dembcat = dhprebn @ W1.T           # (n, n_hidden) @ (n_hidden, n_embd*block_size)
dW1 = embcat.T @ dhprebn           # (n_embd*block_size, n) @ (n, n_hidden)
db1 = dhprebn.sum(0)               # (n_hidden,)
```

```python
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
```

#### 5.2.7 Embedding Table的反向传播

最后一步是从`embcat`反向传播到Embedding查找表`C`。前向传播中，`embcat = emb.view(emb.shape[0], -1)`将`(n, block_size, n_embd)`reshape为`(n, block_size * n_embd)`。反向传播时只需要做相反的reshape操作：

```python
demb = dembcat.view(emb.shape)     # (n, block_size, n_embd)
```

Embedding层的梯度计算有一个独特之处：同一个字符可能在batch中多次出现（不同样本、不同位置可能引用`C`中的同一行），因此`dC`的更新不是赋值，而是**累加**。我们需要遍历batch中的每个位置，将`demb`中对应的梯度向量累加到`dC`的对应行上：

```python
dC = torch.zeros_like(C)           # 初始化全零梯度表
for k in range(Xb.shape[0]):       # 遍历batch中每个样本
    for j in range(Xb.shape[1]):   # 遍历每个上下文位置
        ix = Xb[k, j]              # 获取该位置的字符索引
        dC[ix] += demb[k, j]       # 将梯度累加到对应行
```

这个双重循环是Embedding层反向传播的标准实现。在大型模型中（如GPT），这个循环会被优化为更高效的scatter-add操作，但逻辑本质相同：同一个embedding向量的梯度是所有引用它的位置的梯度之和。

```python
cmp('emb', demb, emb)
cmp('C', dC, C)
```

至此，我们已经完成了整个网络的手动反向传播。如果所有`cmp()`调用都返回`approximate: True`且`maxdiff`极小，恭喜你——你已经成为了反向传播忍者。

---

### 5.3 完整手动实现

#### 5.3.1 用`cmp()`函数验证数值一致性

在前面的推导中，我们已经展示了如何使用`cmp()`函数逐层验证手动梯度与PyTorch autograd的一致性。这种逐步验证的方法论至关重要：与其一次性写完所有反向传播代码然后发现某个梯度对不上，不如每完成一层就验证一次。当`cmp()`报告某个梯度不匹配时，你知道问题一定出在你刚刚写的那几行代码中，而不是整个反向传播链路。

#### 5.3.2 逐步验证的策略

推荐的验证策略是从损失函数端开始，一步步向前验证。每验证通过一层，就将该层视为"可信的基础设施"，在此基础上构建下一层的反向传播。具体步骤如下：

1. **先验证Loss层**：确保`dlogprobs`正确。这是最简单的部分，如果这里就出错，后续的验证都会被干扰。
2. **再验证Cross Entropy的展开部分**：逐一手动推导`probs`、`counts`、`norm_logits`、`logits`的梯度，每一步都用`cmp()`确认。
3. **验证第2层Linear**：`dW2`、`db2`、`dh`的推导。
4. **验证Tanh**：确认`dhpreact`。
5. **验证BatchNorm**：这是最复杂的部分，如果前面都验证通过了，BatchNorm的错误就相对容易定位。
6. **验证第1层Linear和Embedding**：最后两步相对简单。

这种"从后往前、逐层确认"的策略不仅适用于本课的练习，也是你在实际研究中调试自定义层或新架构时的标准流程。

#### 5.3.3 完整训练循环：手动梯度更新

验证完所有梯度后，我们可以把手动反向传播整合进一个完整的训练循环中。关键的区别在于：我们不再调用`loss.backward()`，而是自己计算所有梯度；同时，我们使用`torch.no_grad()`上下文管理器来告诉PyTorch不需要追踪计算图，这能显著减少内存占用和计算开销。

```python
# 完整训练循环（手动反向传播版本）
n_embd = 10
n_hidden = 200

# 初始化参数
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]

max_steps = 200000
batch_size = 32
n = batch_size
lossi = []

# 关键：使用 torch.no_grad() 包裹整个训练循环
with torch.no_grad():
    for i in range(max_steps):
        # 构建 mini-batch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # ===== 前向传播 =====
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        # ===== 手动反向传播 =====
        # 1) Cross Entropy backward（融合版本，见5.4节）
        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n

        # 2) 第2层线性层
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)

        # 3) Tanh
        dhpreact = (1.0 - h ** 2) * dh

        # 4) BatchNorm（这里用逐步版本展示；融合版本见5.4节）
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dbnraw = bngain * dhpreact
        dbndiff = bnvar_inv * dbnraw
        dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
        dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
        dbndiff2 = (1.0 / (n-1)) * dbnvar
        dbndiff += (2 * bndiff) * dbndiff2
        dhprebn = dbndiff.clone()
        dbnmeani = (-dbndiff).sum(0)
        dhprebn += 1.0 / n * dbnmeani

        # 5) 第1层线性层
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)

        # 6) Embedding层
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix_tok = Xb[k, j]
                dC[ix_tok] += demb[k, j]

        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # ===== 参数更新（用手动梯度替代 p.grad） =====
        lr = 0.1 if i < 100000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        # 跟踪训练进度
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
```

Karpathy在视频中用一个meme形象地对比了两种训练方式：使用`loss.backward()`然后`p.data += -lr * p.grad`的是"cheems doge"（弱小无助），而使用自己手写的反向传播然后`p.data += -lr * grad`的是"swole doge"（强大自信）。这个meme虽然是玩笑，但确实传达了一个深刻的信息：**当你真正理解每一行代码在做什么时，你对模型的掌控感是完全不同的**。

---

### 5.4 融合优化

#### 5.4.1 Cross Entropy的融合反向传播

在5.2.2节中，我们将Cross Entropy Loss拆解为十余个基本操作，逐一推导了每个操作的梯度。这是一个极好的练习，但在实际训练中，我们很少这样一步步展开。原因是：softmax和负对数似然（negative log-likelihood, NLL）的组合有一个极其优雅的数学简化。

回忆Cross Entropy Loss的定义（省略数值稳定性的max减法）：

$$\text{loss} = -\frac{1}{n} \sum_i \log\left(\frac{e^{z_{i,y_i}}}{\sum_j e^{z_{i,j}}}\right)$$

其中$z_i$是第$i$个样本的logits向量，$y_i$是其正确标签。对这个表达式关于logits$z_{i,k}$求导，经过一系列代数运算（ softmax的导数特性使得大部分项相互抵消），可以得到一个惊人的简洁结果：

$$\frac{\partial \text{loss}}{\partial z_{i,k}} = \frac{1}{n} \left( p_{i,k} - \mathbf{1}[k = y_i] \right)$$

其中$p_{i,k} = \text{softmax}(z_i)_k$是模型预测的概率。换句话说：**Cross Entropy对logits的梯度就是softmax概率，在正确标签的位置上减1，然后除以batch size**。

这个简洁形式的代码实现只有三行：

```python
# Cross Entropy 融合反向传播
loss_fast = F.cross_entropy(logits, Yb)   # 用PyTorch内置函数计算loss

dlogits = F.softmax(logits, 1)            # 第一步：softmax概率
dlogits[range(n), Yb] -= 1                # 第二步：正确位置减1
dlogits /= n                              # 第三步：除以batch size

cmp('logits', dlogits, logits)            # 验证：与逐步推导的结果完全一致
```

`dlogits`的每一行之和有什么特点？对于每个样本，softmax概率之和为1，减1后（只在正确位置减），所有元素之和变为$0$。这意味着Cross Entropy的梯度在logits空间中是**零和**的——模型通过提高正确类别的logit、同时降低错误类别的logit来学习，而且提高和降低的总量精确平衡。

这个融合公式是深度学习中最优雅、最常用的梯度公式之一。在PyTorch内部，`F.cross_entropy`的反向传播正是以这种方式高效实现的，避免了构造softmax概率的完整中间张量。

#### 5.4.2 BatchNorm的融合反向传播

与Cross Entropy类似，BatchNorm的多步展开也可以融合为一个紧凑的表达式。给定上游梯度`dhpreact`，我们直接计算`dhprebn`（即经过BatchNorm之前的隐藏层激活的梯度），跳过所有中间变量。

融合公式的推导过程较为复杂，需要对BatchNorm的五个步骤（去均值、计算方差、归一化、缩放、平移）同时进行链式法则展开，并将所有中间项合并化简。最终的结果是：

```python
# BatchNorm 融合反向传播
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact 
    - dhpreact.sum(0) 
    - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
)

cmp('hprebn', dhprebn, hprebn)   # 验证
```

让我们解读这个公式的直觉。`dhprebn`是BatchNorm输入端的梯度，它由三项组成：

- **第一项 `n * dhpreact`**：这是"直通"路径的梯度，如果BatchNorm不存在，梯度就是这样直接传回去的。乘以`n`是因为后面的两项都带有平均化的分母。

- **第二项 `- dhpreact.sum(0)`**：这是对均值操作（`hprebn.mean()`）的梯度补偿。BatchNorm减去了batch均值，这意味着所有样本的梯度都通过均值计算相互耦合了。这一项抵消了"减去均值"操作引入的梯度流动。

- **第三项 `- n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)`**：这是对方差操作（`hprebn.var()`）的梯度补偿。BatchNorm除以了标准差，而标准差依赖于batch中所有样本的值。这一项通过`bnraw`（归一化后的值）来调节每个样本的梯度，考虑了方差归一化对梯度流的影响。

在完整的训练循环中，使用融合版本可以显著简化代码并略微提升计算效率。更重要的是，它展示了反向传播中一个普遍的原则：**多个前向操作可以被融合为一个等价的反向操作，而理解这种融合需要对每一层的数学本质有深入的把握**。

#### 5.4.3 融合版训练循环

将Cross Entropy和BatchNorm的融合版本整合在一起，训练循环的反向传播部分变得非常简洁：

```python
# ===== 融合版手动反向传播 =====

# 1) Cross Entropy backward（融合）
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n

# 2) 第2层Linear
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

# 3) Tanh
dhpreact = (1.0 - h ** 2) * dh

# 4) BatchNorm backward（融合版）
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact - dhpreact.sum(0) 
    - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
)

# 5) 第1层Linear
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)

# 6) Embedding
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix_tok = Xb[k, j]
        dC[ix_tok] += demb[k, j]

grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
```

这个融合版的反向传播与5.3.3节的逐步版本在数学上完全等价，但代码更简洁、执行更高效。PyTorch的`F.cross_entropy`和`F.batch_norm`内部实现本质上就是这样的融合逻辑。

---

### 5.5 课后练习与推荐资源

#### 5.5.1 课后练习

以下是Karpathy在视频中布置的四个Exercise，附带详细的解答思路。

**Exercise 1：完整手动反向传播**

本节的核心练习。不看任何提示，独立地从`loss`开始，一步步反向传播通过Cross Entropy Loss、第2层Linear、Tanh、BatchNorm、第1层Linear和Embedding层，为每个变量计算梯度，并用`cmp()`验证。如果某个变量的梯度不匹配，仔细检查前向传播中该变量的生成方式，确认你没有遗漏广播（broadcasting）操作带来的`sum`累加。这是本系列课程中最具挑战性但也最有价值的练习之一。

**Exercise 2：Cross Entropy融合反向传播**

理解为什么`dlogits = F.softmax(logits, 1)`后在正确类别位置上减1再除以`n`，就能得到与逐步推导完全一致的结果。尝试从数学上推导这个公式：写出$\text{loss} = -\frac{1}{n} \sum_i \log(\text{softmax}(z_{i, y_i}))$，对$z_{i,k}$求偏导，利用softmax的导数特性$\partial \text{softmax}_k / \partial z_j = \text{softmax}_k (\delta_{kj} - \text{softmax}_j)$，证明最终结果确实是`softmax(logits) - one_hot(Yb)`的形式。

**Exercise 3：BatchNorm融合反向传播**

推导BatchNorm的融合反向传播公式。从BatchNorm的五个前向步骤出发，将$\hat{x} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$对$x$求导，其中$\mu = \frac{1}{n}\sum x_i$，$\sigma^2 = \frac{1}{n-1}\sum (x_i - \mu)^2$。将所有中间项展开并合并化简，最终应得到5.4.2节中的三行公式形式。验证时确保`cmp('hprebn', dhprebn, hprebn)`通过。

**Exercise 4：用你自己的反向传播训练网络**

将Exercise 1或Exercise 3的反向传播代码整合进完整的训练循环中，使用`torch.no_grad()`上下文管理器，用你自己计算的梯度替代`loss.backward()`。观察训练损失曲线是否与使用PyTorch autograd的版本完全一致（应该在浮点误差范围内相同）。尝试不同的网络宽度（`n_hidden`）和学习率，感受手动梯度更新的灵活性。最后，在训练结束后校准BatchNorm的均值和方差，评估训练集和验证集的损失。

#### 5.5.2 推荐学习资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 本课YouTube视频 | https://www.youtube.com/watch?v=q8SA3rM6ckI | Karpathy原版讲解，时长约2小时 |
| 本课GitHub Notebook | https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb | 完整代码，可直接运行 |
| Google Colab练习 | https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing | 在线练习环境 |
| "Yes you should understand backprop" | https://karpathy.ai/yes-you-should-understand-backpropagation.html | Karpathy的博客文章，阐述为什么理解反向传播至关重要 |
| BatchNorm原始论文 | https://arxiv.org/abs/1502.03167 | Ioffe & Szegedy, 2015[^2^] |
| Bessel's Correction | http://math.oxford.emory.edu/site/math117/besselCorrection/ | 解释样本方差除以$n-1$的统计原理 |
| Bengio et al. (2003) | https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf | 神经网络语言模型的开创性工作[^3^] |

完成本课后，你对神经网络的理解应该达到了一个全新的深度。你不再只是调用`loss.backward()`然后祈祷它工作正确——你知道每个梯度是如何计算出来的，每一条公式背后的直觉是什么。这种对底层机制的把握，将在你面对新架构、新损失函数或训练异常时，成为最有价值的武器。

> **脚注引用**：
> [^1^] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
> [^2^] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
> [^3^] Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). A neural probabilistic language model. *JMLR*, 3, 1137-1155.



---

## 6. 第6课：makemore Part 5 — 构建WaveNet

在上一课中，我们通过手动实现反向传播，对神经网络内部梯度的流动方式建立了深刻的直觉。那些关于链式法则、梯度累积和BatchNorm反向传播的练习，让我们从"仅仅调用`loss.backward()`"的使用者，变成了真正理解底层机制的实践者。但这一路上我们始终使用着一个相对简单的架构 —— 一个两层的多层感知机（Multi-Layer Perceptron, MLP），它将输入字符的embedding拼接后一次性送入隐藏层。这种"扁平"（flat）的处理方式在面对更长的上下文时会遇到一个根本性的瓶颈：参数量随上下文长度线性增长，而且所有信息在网络的入口处就被压成一团，没有机会在层次化的抽象级别上逐步整合。

这一课，我们将解决这个问题。我们会把原来的MLP改造为一种层次化的树状结构，这种结构正是DeepMind在2016年提出的WaveNet的核心思想[^1^]。我们会一步步搭建这个架构，亲眼见证一个简单的想法 —— "逐步合并相邻的表示" —— 如何优雅地解决了长上下文的问题。与此同时，我们还将深入理解PyTorch的`torch.nn`模块内部是如何工作的，亲手实现一个简化版的神经网络层系统；学习在深度学习开发中追踪张量形状（tensor shape tracking）这项最核心的调试技能；体会从Jupyter Notebook实验代码到结构化`.py`生产代码的演进过程；并最终明白为什么这种层次化结构其实等价于一维卷积神经网络。

### 6.1 从全连接到层次化结构

#### 6.1.1 MLP的局限：上下文窗口增大时的参数爆炸

让我们先回顾一下之前MLP的结构，以便清楚地看到问题所在。当`block_size=3`时，我们有3个输入字符，每个字符通过Embedding层被映射为一个`n_embd=10`维的向量。然后这3个向量被拼接（concatenate）成一个长度为`3 * 10 = 30`的向量，传入第一个线性层：

$$h = \tanh(W_1 \cdot [e_1; e_2; e_3] + b_1)$$

这里的分号`[;]`表示向量拼接操作。`W_1`的形状为`(n_embd * block_size, n_hidden)`，即`(30, 200)`，包含6000个参数。对于3个字符的上下文，6000个参数的第一层权重是完全可接受的。网络的第一层同时"看到"了所有3个字符的embedding，并在隐藏层中将它们混合。这个设计的核心问题在于它的**可扩展性**（scalability）—— 当上下文窗口增大时，第一层的参数量会爆炸性增长。

现在考虑我们想要捕捉更长的依赖关系的情况。语言中存在很多需要长上下文才能判断的模式 —— 比如一个名字以"qu"开头，接下来几乎必定是元音"a"；某些前缀（如"br"、"sh"、"th"）会以高度可预测的方式影响后续多个字符的选择概率；甚至有些更长的模式，如"christ-"前缀倾向于以"-ina"或"-opher"结尾。如果我们将上下文从3个字符增加到8个字符，`W_1`的输入维度从30变为80。假设隐藏层仍然保持200个神经元，`W_1`的参数量从6000跃升到16000。如果更进一步，处理16个字符的上下文，`W_1`就有32000个参数，而且这仅仅是网络的第一层。这种增长是不可持续的。

更重要的是，这种"扁平化"（flattening）设计在直觉上也不够优雅。想象一下你阅读一个中文句子时理解其含义的认知过程：你不会把一句话中的8个字同时不经任何预处理地塞进大脑的一个单一处理单元。相反，你会先在较低的层次上识别出词组（比如"深度"、"学习"），然后在更高的层次上理解这些词组之间的关系（"深度学习"是一个领域名称），最终整合为对整个句子的理解。信息处理天然是层次化的（hierarchical） —— 从局部到全局，从具体到抽象，每一层都在前一层的基础上构建更高阶的表示。扁平化的MLP忽略了这种天然的层次结构。

#### 6.1.2 WaveNet的核心思想：用树状结构逐步聚合信息

WaveNet论文的核心理念正是模拟这种层次化的信息处理方式。具体思路非常优雅，可以用一棵二叉树来直观地理解：

假设我们有8个输入位置（`block_size=8`），分别标记为位置1到位置8，每个位置有一个embedding向量。与其在第一步就把全部8个向量首尾相接拼成一个超长向量，不如这样处理：

- **第1层聚合**：将相邻的2个embedding向量拼接起来，通过一个线性层和非线性激活函数，得到第1层的表示。这样原来的8个位置被合并为4组 —— 组A（位置1+2）、组B（位置3+4）、组C（位置5+6）、组D（位置7+8）。每组是一个新的、更高层次的向量表示，可以被认为是"二元字符组"（bigram-level）的特征。
- **第2层聚合**：将相邻的2组表示再次拼接、变换。4组被进一步合并为2组 —— 组AB（组A+组B，覆盖了原始位置1-4）、组CD（组C+组D，覆盖了原始位置5-8）。这相当于"四元字符组"（4-gram level）的抽象。
- **第3层聚合**：继续合并，2组最终变成1组 —— 组ABCD（覆盖了全部8个原始位置）。这个单一的向量表示融合了全部上下文信息，是整个上下文的高层摘要。
- **输出层**：基于这个最终的聚合表示，预测下一个字符的概率分布。

这种结构的优美之处在于多个层面。首先，每一层的线性层处理的输入维度是恒定的 —— 无论总上下文长度是8、16还是32，每一层只是合并两个相邻的表示，所以`Linear`的输入始终是前一层输出维度的2倍（因为FlattenConsecutive(2)拼接了两个向量），输出维度保持为`n_hidden`。其次，感受野（receptive field）随着层数**指数级增长**：每向下走一层，每个输出位置能"看到"的原始输入范围就翻倍。3层可以覆盖$2^3=8$个位置，4层覆盖$2^4=16$个，5层覆盖$2^5=32$个。这与扁平MLP中感受野随参数线性增长形成鲜明对比 —— 在扁平MLP中，将感受野从8扩大到16需要第一层的参数量翻倍，而在层次化结构中只需增加一层，参数量仅线性增加。

从参数效率的角度看，层次化结构也更优。虽然网络变得更深（更多的层数），但每一层的宽度可以控制得很合理。信息通过深度而非宽度来流动，这正符合现代深度学习架构设计的主流哲学。更深但更窄的网络往往比更浅但更宽的网络有更好的归纳偏置（inductive bias），因为层次化的特征提取天然与许多现实世界问题（如语言、图像、音频）的结构相吻合。每一层都在学习一种更高阶的抽象：第1层可能学到哪些字符经常相邻出现，第2层学到哪些字符组合经常共现，第3层则学到整个名字片段的语义模式。

#### 6.1.3 WaveNet论文简介：原始音频生成的深度自回归模型

WaveNet是DeepMind团队在2016年发表的一篇里程碑式论文[^1^]，其原始目标是生成原始音频波形（raw audio waveform）。与我们现在处理离散字符的文本生成任务不同，WaveNet预测的是连续的（或经过$\mu$-law量化的）音频振幅值。尽管目标领域不同，但其底层架构思想 —— 层次化地聚合历史信息 —— 是完全通用的，可以迁移到任何自回归序列建模任务中。

WaveNet是一个**自回归**（autoregressive）且**全概率**（fully probabilistic）的生成模型。"自回归"意味着每个时间步的预测都依赖于之前所有的输出；"全概率"意味着它对每个样本都建模一个完整的条件概率分布（通常用混合分布或离散分类分布），而非仅预测一个点估计。在原始论文中，WaveNet在文本转语音（Text-to-Speech, TTS）任务上达到了当时最先进的效果，生成的语音质量之高清令许多评论者难以区分合成语音与真人录音。更有趣的是，WaveNet能够通过条件输入（conditioning input）模拟多个不同的说话人声音，甚至可以生成音乐片段。这些成就的背后，正是层次化树状结构提供的强大建模能力 —— 它可以捕捉到音频信号中从毫秒级的波形模式到秒级的语义模式的多个时间尺度的依赖关系。

原始WaveNet使用**扩张因果卷积**（dilated causal convolution）来高效地实现我们上面描述的层次化结构。"因果"（causal）是一个关键约束，它保证模型在预测第$t$个时间步时只能使用$1, 2, ..., t-1$时刻的信息，不能使用任何"未来"的信息 —— 这对于自回归生成是核心要求。"扩张"（dilated）则是实现高效层次化聚合的关键技巧：通过让卷积核在连续的层中逐步跳过更多的时间步，感受野可以呈指数级增长，而参数量仅线性增长。本课我们先用一种更朴素但更易理解的方式 —— 通过自定义的`FlattenConsecutive`层显式地重组张量形状 —— 来模拟相同的层次化聚合逻辑，这样可以清楚地看到数据是如何逐层被聚合的。

### 6.2 架构设计

#### 6.2.1 FlattenConsecutive层：将相邻的embedding向量拼接

`FlattenConsecutive`是本课架构创新的核心组件。它的概念很简单但实现很关键：给定一个形状为`(B, T, C)`的三维张量 —— 分别代表batch大小（Batch size）、时间步数（Time steps，即序列长度）和特征维度（Channels，即特征数或通道数）— 它将每`n`个连续的时间步在特征维度上拼接在一起，输出形状变为`(B, T//n, C*n)`。当合并后只剩一个时间步时，它会自动squeeze掉时间维度，变为2D张量。

让我们仔细看看实现：

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n  # 每n个连续时间步进行合并

    def __call__(self, x):
        B, T, C = x.shape
        # 关键操作：将时间维度每n个一组，在特征维度上拼接
        # 等价于先把 (B, T, C) 重塑为 (B, T//n, n, C) 然后合并最后两维
        x = x.view(B, T // self.n, C * self.n)
        # 如果合并后只剩一个时间步，去掉该维度，变回2D张量
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []  # 该层没有可训练参数
```

这里最关键的操作是`view(B, T // self.n, C * self.n)`。我们可以通过一个具体例子来逐步理解：假设输入`x`的形状是`(4, 8, 24)`，表示batch=4，8个时间步，每步24维特征。经过`FlattenConsecutive(2)`处理后，形状变成`(4, 4, 48)`。这意味着原来的8个时间步被分成了4对，每对的48维向量是原来两个24维向量的拼接。如果再经过一次`FlattenConsecutive(2)`，形状变为`(4, 2, 96)` —— 4个组被分成2对，每对96维。第三次经过`FlattenConsecutive(2)`，形状变为`(4, 1, 192)`，然后`squeeze(1)`将其变为`(4, 192)`。这个形状变化的过程清晰地展现了信息是如何逐层被聚合的。

这个层没有可训练参数，它的角色纯粹是**以语义上有意义的方式改变张量形状** —— 把相邻的位置在特征维度上"绑"在一起，为下一层线性变换做准备。可以把它理解为一种"手工设计的注意力模式"：我们显式地告诉网络哪些位置应该先被放在一起处理。随着数据在网络中向下流动，被"绑定"在一起的区域越来越大，从相邻字符到相邻词组再到整个短语，最终形成对整个上下文的理解。这种渐进式的信息聚合方式比一次性展平要自然得多，也更符合人类处理信息的方式。

#### 6.2.2 树状结构的构建：逐步增大感受野

有了`FlattenConsecutive`，我们就可以搭建完整的层次化网络了。为了对比，让我们先看之前的"扁平"方式 —— 一次性把所有8个字符的embedding拼接成一个长向量：

```python
# 扁平结构（传统MLP方式）— 用于对比：
# model = Sequential([
#   Embedding(vocab_size, n_embd),
#   FlattenConsecutive(8),         # 一次性展平所有8个字符
#   Linear(n_embd * 8, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#   Linear(           n_hidden, vocab_size),
# ])
```

这种方式的问题在于`Linear(n_embd * 8, n_hidden)`这一层的输入维度太大。当`n_embd=24`时，输入是192维，即使`n_hidden=128`，这一层也有`192 * 128 = 24576`个参数。而且所有信息在这里一次性混合，前面的层没有任何机会做预处理和特征提取。直觉上，这是一种浪费：网络的第一层被迫同时处理低级的字符级别特征和高级的语义级特征，没有层次化的抽象过程。

现在来看WaveNet风格的层次化结构：

```python
n_embd = 24      # 字符嵌入维度
n_hidden = 128   # 每个隐藏层的维度

model = Sequential([
    Embedding(vocab_size, n_embd),
    # ========== Layer 1: 底层聚合 ==========
    # 每2个embedding拼接 -> 8个位置合并为4组
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    # ========== Layer 2: 中层聚合 ==========
    # 每2组拼接 -> 4组合并为2组
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    # ========== Layer 3: 顶层聚合 ==========
    # 每2组拼接 -> 2组合并为1组
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    # ========== 输出层 ==========
    Linear(n_hidden, vocab_size),
])

# 最后一层初始化：降低初始输出幅度，使训练更稳定
with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))  # 输出: 约76000
for p in parameters:
    p.requires_grad = True
```

让我们仔细追踪数据流经这个网络时张量形状的变化，假设`batch_size=4`，`block_size=8`：

| 层 | 操作 | 输出形状 | 说明 |
|---|---|---|---|
| Embedding | 查表 | `(4, 8, 24)` | 8个字符，每字符24维embedding |
| FlattenConsecutive(2) | 拼接相邻2个 | `(4, 4, 48)` | 8个位置 -> 4组，每组48维 |
| Linear | 线性变换 | `(4, 4, 128)` | 48维 -> 128维，独立的4组表示 |
| BatchNorm1d + Tanh | 归一化+激活 | `(4, 4, 128)` | 4组，每组128维，已非线性变换 |
| FlattenConsecutive(2) | 拼接相邻2组 | `(4, 2, 256)` | 4组 -> 2组，每组256维 |
| Linear | 线性变换 | `(4, 2, 128)` | 256维 -> 128维，更高层特征 |
| BatchNorm1d + Tanh | 归一化+激活 | `(4, 2, 128)` | 2组，更抽象的表示 |
| FlattenConsecutive(2) | 拼接相邻2组 | `(4, 192)` | 2组 -> 1组，squeeze后为2D |
| Linear | 线性变换 | `(4, 128)` | 192维 -> 128维 |
| BatchNorm1d + Tanh | 归一化+激活 | `(4, 128)` | 最终的聚合表示，融合全部上下文 |
| Linear | 输出层 | `(4, 27)` | 27个字符的logits |

注意到每一层的`Linear`接收的输入维度恰好是前一层输出维度的两倍（因为`FlattenConsecutive(2)`把两个向量拼接在一起），而输出维度保持为`n_hidden=128`。这种固定的模式让网络可以自然地加深 —— 想要覆盖16个字符的上下文？只需在第3层后再加一组`FlattenConsecutive(2) + Linear + BatchNorm + Tanh`。感受野会从8翻倍到16，参数量只增加一个线性层的权重（约$128 \times 256 = 32768$个参数），而非扁平结构中那种与上下文长度成正比的爆炸式增长。这种可扩展性是层次化结构最大的优势。

为了更直观地理解这个层次化结构的数据流，让我们用一个具体的例子来追踪。假设输入的8个字符是"sophia.."（名字后面用'.'填充到8位），每个字符被嵌入为一个24维向量。在Embedding层之后，我们有8个并行的向量，分别对应位置1到位置8。第1层的`FlattenConsecutive(2)`将它们两两配对：位置1+2（"so"的联合表示）、位置3+4（"ph"）、位置5+6（"ia"）、位置7+8（".."）。每对经过线性层和tanh后，变成4个128维的向量。

第2层将这4个向量再两两配对：(组1+组2)的联合表示和(组3+组4)的联合表示。注意此时每个向量已经不再是原始字符的简单拼接，而是经过了一层非线性变换后的"特征" —— 网络已经学会了"so"和"ph"各自的某种抽象模式。第2层的线性层将这些更高阶的特征再次混合，产出2个128维向量。

第3层将最后2个向量合并为一个，它承载了从位置1到位置8的全部信息。但这个信息已经经过了3轮非线性变换和特征提取，远比简单拼接192个数字要丰富得多。这就是深度网络的力量：每一层都在前一层的基础上构建更高阶的表示，从原始embedding到局部特征到全局特征，逐步抽象。

让我们更进一步思考为什么这种树状结构比扁平结构更好。想象你要预测名字"christopher"中的下一个字符。当你看到前8个字符"christop"时，"christ-"这个前缀已经给了很强的信号 —— 这个名字很可能以"-opher"结尾。在扁平的MLP中，网络需要在第一层就同时处理"c"的embedding和"h"的embedding和"r"的embedding...一直到"p"的embedding，在192维的空间中一次性找出"christ"这个模式。而在层次化结构中，第1层先学会了"ch"、"ri"、"st"、"op"这些局部模式，第2层将这些组合成更高阶的特征，第3层最终确认了整个"christop"片段的语义。每一层只负责比自己上一层稍大一点的模式，这种分而治之的策略让学习变得更容易。

这与计算机视觉中的卷积神经网络（CNN）有异曲同工之妙。在CNN中，浅层卷积核检测边缘和纹理，中层检测形状和部件，深层检测完整的物体。在我们的字符级语言模型中，浅层检测相邻字符的关系，中层检测音节级的模式，深层检测整个名字片段的语义。这种从局部到全局的特征层次是深度网络强大表达能力的根本来源。

#### 6.2.3 与卷积的关系：这种结构等价于dilated causal convolution

课上Karpathy强调了一个非常重要的洞察：我们在上面用`FlattenConsecutive`和`Linear`层所实现的效果，其实可以用**一维卷积**（1D convolution）更高效、更优雅地完成[^1^]。

具体来说，每一层的`Linear`实际上是在对序列的每个位置独立地应用相同的线性变换。当`Linear(n_hidden * 2, n_hidden)`接收到形状为`(B, T, n_hidden * 2)`的输入时，它会对时间维度上的每个位置`t`执行相同的矩阵乘法 —— 这正是一个卷积核在序列上滑动的行为。卷积核的权重就是`Linear`的权重矩阵，它在每个位置接收两个相邻的输入特征块，产出一个输出特征向量。唯一的区别在于，我们的`FlattenConsecutive`显式地重排了张量形状，而卷积通过核的滑动隐式地做到了同样的事情。

原始WaveNet论文正是用**扩张因果卷积**来实现这种结构的。"因果"约束保证了模型不会"偷看"未来的信息，这对于自回归生成是核心要求。具体来说，"因果卷积"确保在预测位置$t$的输出时，卷积核只覆盖位置$1, 2, ..., t-1$的输入，从不接触$t$或更后的位置。这通过在输入的左侧做适当的填充（padding）来实现。"扩张"（dilated）则是效率的关键：在第1层，卷积核覆盖相邻位置（dilation rate = 1）；在第2层，卷积核跳过1个位置（dilation rate = 2），直接合并间隔的表示；在第3层，dilation rate = 4，以此类推。感受野随层数指数增长（$1, 2, 4, 8, 16, ...$），而每层只需固定大小的卷积核（通常kernel_size = 2），参数量仅随深度线性增长。

理解这个等价关系非常重要，因为它揭示了一个更普适的原理：**很多看似不同的神经网络操作，本质上都是在以不同的方式重组和混合信息**。`FlattenConsecutive` + `Linear`的组合和`Conv1d`做的是同一件事，只是前者通过显式改变形状来实现，后者通过卷积核的隐式滑动来实现。当我们从`FlattenConsecutive`过渡到`Conv1d`时，我们并不是在改变网络的"思维"方式，只是在改变它的"身体" —— 让它更高效地执行同样的计算。

#### 6.2.4 用for循环理解卷积

为了帮助理解卷积与我们的`FlattenConsecutive`结构的等价关系，让我们做一个思想实验。假设我们有一个简单的序列处理任务：对输入序列的每个位置，用同一个线性层处理该位置及其邻居。用我们的层次化结构，这由`FlattenConsecutive`和`Linear`自动完成。但如果用朴素的for循环来写，会是这样的：

```python
# 朴素实现：用for循环逐个位置处理
# logits = torch.zeros(batch_size, vocab_size)
# for t in range(block_size):
#     logits += some_linear_layer(x[:, t, :])

# 用卷积实现：一次完成所有位置的处理
# logits = conv1d(x)  # 卷积核在序列上滑动
```

卷积的本质就是一个"被编译优化过的for循环"。它在底层可能使用高度并行化的矩阵乘法（GEMM）或专门的卷积算法，但概念上与我们用`FlattenConsecutive`+`Linear`做的事是一样的：在序列的每个位置应用相同的线性变换。卷积的优势在于：(1) GPU上的高度优化实现，(2) 自然的权重共享机制，(3) 通过stride、padding、dilation等超参数灵活控制感受野，而无需改变数据流的逻辑。

理解这一点后，从本课的`FlattenConsecutive`实现过渡到正式的卷积实现就不再是"学习一个全新的概念"，而是"换一种更高效的方式做同样的事"。这正是Karpathy的教学风格 —— 总是从最朴素、最易于理解的实现出发，然后揭示它如何与更高级、更高效的实现相联系。

### 6.3 torch.nn深度使用

#### 6.3.1 自定义nn.Module：封装各层为可复用模块

在这一课之前，我们的网络参数是"裸露"定义的 —— `C`、`W1`、`b1`、`W2`、`b2`等作为独立变量存在，前向传播是一长串操作语句。这种方式对于理解底层机制很有教育意义，但随着架构变复杂，我们需要更好的组织结构来管理越来越多的层和参数。PyTorch的`torch.nn`模块系统正是为此而生。

Karpathy在这一课中做了一个非常有趣且有教育意义的选择：他没有直接使用PyTorch官方的`nn.Module`，而是**自己从头实现了一个简化版本**。为什么要多此一举？因为通过亲手实现这些我们每天都在使用的抽象，我们能真正理解它们背后的设计哲学和运行机制。当你在使用`nn.Linear`或`nn.BatchNorm1d`时，如果你知道它们内部只有几十行代码，并且你自己也能写出来，你对整个框架的信心和使用能力都会完全不同。这种"不迷信框架，理解其本质"的态度是成为优秀深度学习工程师的关键。

让我们逐一看看实现的关键层。

**Linear层**承载了网络中的主要可训练参数：

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming初始化：权重除以sqrt(fan_in)，保持前向传播方差稳定
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

`Linear`层封装了一个权重矩阵`W`和一个可选的偏置向量`b`，执行的操作是`y = x @ W + b`。初始化时使用的Kaiming初始化（除以$\sqrt{fan\_in}$）来自我们在第4课中详细推导过的原理 —— 它保证了无论输入特征维度多大，经过矩阵乘法后输出值的方差大致保持在1左右，从而避免信号在深层网络中指数级衰减或放大。

**BatchNorm1d层**是我们实现中最复杂的组件，尤其是它需要支持从2D到3D输入的切换：

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps           # 防止除零的小常数
        self.momentum = momentum  # 运行时统计量的更新动量
        self.training = True     # 训练/推理模式切换标志
        # 可学习的缩放和平移参数
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # 运行时统计量（在推理时使用，不通过梯度更新）
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # 关键：根据输入维度决定沿哪些轴计算统计量
            if x.ndim == 2:
                dim = 0  # (B, C)：沿batch维度
            elif x.ndim == 3:
                dim = (0, 1)  # (B, T, C)：同时沿batch和时间维度
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            # 推理阶段：使用训练期间累积的运行时统计量
            xmean = self.running_mean
            xvar = self.running_var
        # 标准化：减去均值，除以标准差
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # 缩放和平移：让网络学习恢复可能有用的原始尺度
        self.out = self.gamma * xhat + self.beta
        # 更新运行时统计量（仅在训练时且使用no_grad）
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

这个实现有一个至关重要的细节：当输入从2D `(B, C)` 变为3D `(B, T, C)` 时，均值和方差的计算维度必须相应调整。对于3D输入，我们同时沿batch维度（0）和时间维度（1）做reduce，这样每个通道（channel）只有一个均值和一个方差。如果错误地只沿`dim=0`计算，那么每个时间步会有独立的统计量 —— 这完全违背了BatchNorm的设计意图，因为这相当于对序列的不同位置应用了不同的归一化，破坏了卷积核权重共享的核心假设。正确的做法是"把batch和时间维度都展平"，把所有样本的所有时间步放在一起计算每个通道的统计量，使得同一个通道在不同时间步上被一致地归一化。这个bug非常隐蔽，因为代码不会报错，模型也能训练，只是性能下降 —— 这种"静默失败"是深度学习中最难调试的问题类型。

**Embedding层**是最简单的可训练层之一，但也是最常用的：

```python
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]
```

它本质上就是一个可训练的查找表（lookup table）。输入是整数索引张量（字符ID），输出是对应行的embedding向量。`self.weight[IX]`利用了PyTorch的高级索引（fancy indexing），可以直接用一个整数张量来索引参数矩阵的多行，返回对应的嵌入向量。这个操作在底层极其高效，因为它本质上只是内存的拷贝，不涉及任何计算。

**Tanh激活层**：

```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

激活函数层没有可训练参数，返回空列表。但它在`__call__`中将输出保存到`self.out`，这个设计在调试时非常有用 —— 我们可以随时检查任何一层的输出值分布，判断tanh是否饱和（大量输出值接近-1或1），这对诊断梯度消失问题至关重要。一个健康的训练过程中，tanh输出的直方图应该主要集中在0附近，呈钟形分布，只有少量尾部延伸到接近-1或1的区域。

这些自定义层的共同设计模式是什么？每个层都实现了三个标准接口：

1. **`__init__`**：初始化参数（如果有）和配置。
2. **`__call__`**：定义前向传播的具体计算逻辑，并将输出保存到`self.out`便于调试。
3. **`parameters()`**：返回该层所有可训练参数的列表，没有参数的层返回空列表。

这个简单的三接口契约正是PyTorch `nn.Module`的核心设计哲学。通过亲手实现它，我们理解了为什么PyTorch要这样设计 —— 统一的接口让不同的层可以以完全一致的方式被组合、遍历和优化，这是整个深度学习框架可扩展性的基石。当你以后使用更复杂的层（如LSTM、Transformer Block、Attention）时，它们也都遵循同样的接口契约。

#### 6.3.2 Sequential容器：简化前向传播代码

当网络包含十几层时，手动逐层调用前向传播会变得冗长且容易出错。`Sequential`容器提供了一个优雅的解决方案：按顺序执行一组层，让前向传播代码变得简洁而清晰。

```python
class Sequential:
    def __init__(self, layers):
        self.layers = layers  # 层列表

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # 嵌套列表推导：遍历所有层，收集每层的参数，展平为一个列表
        return [p for layer in self.layers for p in layer.parameters()]
```

`parameters()`方法虽然只是一行代码，但值得仔细品味。它使用嵌套列表推导式遍历`self.layers`中的每个层，再对每个层调用`layer.parameters()`收集其参数，最终将所有参数展平到一个单一的列表中。这使得我们可以用一行代码获取模型的全部参数：`parameters = model.parameters()`，然后统一设置`requires_grad=True`，在训练循环中统一进行参数更新。这种参数收集的递归模式（容器收集其子元素的参数）是`nn.Module`系统最强大的设计之一。它让任意复杂的嵌套结构（比如一个Sequential里面包含另一个Sequential，或者包含自定义的复杂模块）都能自动且正确地收集所有参数。

有了`Sequential`，训练循环变得非常优雅：

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # 构建mini-batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # 前向传播：一行代码完成所有层的计算
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()

    # 参数更新
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪训练进度
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
```

注意这里我们又回到了使用PyTorch自动求导（`loss.backward()`），而非上一课的手动反向传播。在快速迭代和实验架构设计的阶段，使用自动求导可以显著提高开发速度。手动反向传播的技能在我们需要调试梯度问题、实现自定义梯度操作，或者将模型部署到不支持autograd的环境中时才会真正派上用场。这两种技能不是互斥的，而是互补的 —— 理解手动反向传播让你能更好地使用自动求导，而自动求导让你能快速实验复杂的架构。一个优秀的深度学习工程师应当同时具备这两种能力，并在不同的场景下灵活切换。

#### 6.3.3 参数初始化：不同的初始化策略对比

初始化策略对训练的稳定性有着决定性的影响。在这一课的网络中，我们在不同位置使用了不同的初始化方式，每种都有其明确的理论或经验依据。

**Linear层的权重使用Kaiming初始化**：
`self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5`。这保证了输入输出方差的一致性。回顾一下我们在第4课中的推导：假设输入$x$的各个元素是独立同分布、均值为0、方差为1的随机变量，那么经过$y = x @ W$后，$y$的每个元素的方差约为$fan\_in \cdot Var(x_i) \cdot Var(W_{ij}) = fan\_in \cdot 1 \cdot (1/fan\_in) = 1$。由于我们还使用了tanh激活函数，而tanh在0附近的斜率接近1，这个方差可以相对稳定地传递多层。如果没有这个初始化因子，当`fan_in`很大时（比如192），未经缩放的随机权重会让输出方差爆炸，导致tanh立即饱和，梯度消失，训练根本无法开始。

**最后一层输出层使用缩小的初始化**：
`model.layers[-1].weight *= 0.1`。这是一个非常实用的小技巧。在训练开始时，我们希望模型处于一种"无知"的状态 —— 对于27个可能的字符，初始预测应该接近均匀分布（每个字符的概率约为$1/27 \approx 0.037$）。如果最后一层输出的logits值很大，经过softmax后会变得极其"尖锐"：一个概率接近1，其他全部接近0。这意味着模型在训练初期就过于自信，损失函数的梯度也会非常大，可能导致训练不稳定甚至发散。通过将最后一层权重缩小到原来的0.1倍，初始的logits值较小（接近0），softmax输出更加平滑均匀，损失函数处于一个更温和的区域，优化过程可以更稳定地开始。这种"让模型从谦逊开始"的初始化策略在实际训练大型网络时尤为重要。一些现代架构（如Transformer）甚至会使用更精细的初始化公式，比如根据层深度来调整初始化的尺度。

#### 6.3.4 自定义层与PyTorch生态的兼容性

一个自然的问题是：我们自己实现的这些简化层（`Linear`、`BatchNorm1d`、`Embedding`等）能否与PyTorch官方的`nn.Module`生态系统混用？答案是肯定的，只要你的自定义层遵循相同的接口契约（`__init__`、`__call__`或`forward`、`parameters`），它们就可以无缝地与官方组件组合。

例如，你可以创建一个混合模型：

```python
# 混合使用自定义层和PyTorch官方层
model = Sequential([
    Embedding(vocab_size, n_embd),           # 我们的自定义层
    FlattenConsecutive(2),
    torch.nn.Linear(n_embd * 2, n_hidden),   # PyTorch官方层
    BatchNorm1d(n_hidden),                    # 我们的自定义层
    torch.nn.Tanh(),                          # PyTorch官方层
    torch.nn.Linear(n_hidden, vocab_size),
])
```

这种混用是完全可行的，因为PyTorch的`nn.Linear`也实现了`parameters()`方法（返回`[self.weight, self.bias]`），我们的`Sequential`的`parameters()`方法通过遍历所有层并调用各自的`parameters()`来收集参数，不在乎每个层是来自哪里。这种**接口驱动的兼容性**是良好软件设计的典范，也是PyTorch生态系统能够如此丰富的原因 —— 任何遵循`nn.Module`接口的第三方库都可以无缝集成。

### 6.4 性能优化与调试技巧

#### 6.4.1 Tensor形状追踪：深度学习中最重要的调试技能

在这一课的开发过程中，Karpathy反复向我们演示了一项核心技能：**追踪张量形状**（tensor shape tracking）。当网络变深，尤其是引入了`FlattenConsecutive`这种会改变张量维数的层之后，数据在每一层都在变换形状。如果你不能清楚地知道每个中间张量的形状，几乎不可能写出正确的代码，更不可能调试它。

最佳实践是在每一层之后检查输出形状。由于我们在每个自定义层的`__call__`中都将输出保存到`self.out`，调试时可以方便地检查：

```python
# 在训练循环中检查每一层的输出形状
logits = model(Xb)
for i, layer in enumerate(model.layers):
    print(f"Layer {i:2d} ({layer.__class__.__name__:20s}): "
          f"output shape = {tuple(layer.out.shape)}")
```

这一课中一个真实发生的bug完美说明了形状追踪的重要性。在最初的BatchNorm实现中，当输入从2D变为3D时，如果错误地只沿batch维度（`dim=0`）计算均值和方差，而不是同时沿batch和时间维度（`dim=(0, 1)`），那么每个时间步就会有独立的统计量。这个bug不会导致程序崩溃 —— 张量形状在数学运算上仍然完全兼容 —— 但模型的性能会明显下降。只有通过仔细检查中间输出的统计行为，比如打印`xmean.shape`并与预期对比（错误实现会输出`(1, T, C)`而非正确的`(1, 1, C)`），才能发现这个隐蔽的问题。这类"静默失败"的bug是深度学习中最难调试的类型，因为一切看起来都在工作，只是效果不好。

除了形状追踪，另一个极其有用的调试技巧是**激活值分布检查**。在训练过程中，你可以定期打印各层激活值的统计信息：

```python
# 在训练循环中检查激活值分布
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'out') and layer.out is not None:
        print(f"Layer {i} ({layer.__class__.__name__}): "
              f"mean={layer.out.mean():.3f}, std={layer.out.std():.3f}, "
              f"saturation={(layer.out.abs() > 0.97).float().mean():.3f}")
```

对于tanh层，"saturation"（饱和率）是一个关键指标。如果超过50%的神经元输出接近-1或1，说明tanh严重饱和，梯度会非常小，学习几乎停滞。理想情况下，饱和率应低于20%。如果饱和率过高，说明初始化可能有问题（权重太大），或者学习率太高导致参数跳到了饱和区。

还有一个技巧是**梯度分布检查**。在反向传播之后，你可以检查各层梯度的幅度。如果某些层的梯度几乎为零（vanishing gradient），而其他层的梯度非常大（exploding gradient），这就是梯度流不平衡的信号，可能需要调整架构或初始化。在我们的层次化结构中，由于使用了BatchNorm和Kaiming初始化，梯度流通常是比较健康的 —— 这也是BatchNorm最重要的贡献之一：它稳定了深层网络中的梯度传播。

另一个形状追踪的实战技巧是：当你在写一个复杂的模型时，不要试图一次性把所有层都写对。先写一个最简版本（比如只包含第一层），用一小批数据跑通，打印每一层的输入输出形状，确认符合预期后再添加下一层。这种增量式开发比一次性写完整网络然后面对一堆形状不匹配的错误要高效得多。记住：在深度学习中，**张量形状就是类型系统** —— 如果你知道每个张量的形状，你就已经理解了一半的网络逻辑。另一个有用的习惯是在代码注释中标注每个张量的预期形状，比如`# x: (B, T, C)`，这样当你回头看代码时可以快速理解数据流。

#### 6.4.2 从Jupyter到.py文件：代码重构的最佳实践

这一课也体现了典型的深度学习开发流程的演进。在项目的早期探索阶段，Jupyter Notebook是绝佳的工具 —— 可以随意执行单个单元格、立即查看中间结果、快速绘制loss曲线、尝试不同的超参数、可视化embedding空间。但当架构逐渐稳定，代码变得越来越长时，就需要将其迁移到结构化的`.py`文件中。

迁移到Python脚本的好处是多方面的：

- **版本控制**：`.py`文件用git管理时diff更清晰，代码审查（code review）更容易进行。
- **模块化**：可以将层定义（如`FlattenConsecutive`、`BatchNorm1d`）、模型构建函数、训练循环、数据加载和评估逻辑分别放在不同的模块中，形成清晰的代码结构。
- **可测试性**：可以为各个组件编写独立的单元测试，比如验证BatchNorm在训练/推理模式下的行为是否正确，验证FlattenConsecutive对边界情况（如T不能被n整除）的处理。
- **可复现性**：脚本可以通过命令行参数运行，更容易记录和复现实验配置，这对于学术研究和工程实践都至关重要。
- **协作**：团队成员更容易理解和修改结构化的Python代码，而不是一个几百行、执行顺序混乱的Notebook。

Karpathy的建议是：在Notebook中快速探索想法，验证后再及时重构到`.py`文件中。不要让你的"实验代码"无限膨胀。当一个Notebook超过几百行，或者你开始复制粘贴大段代码来尝试变体时，就是重构的信号。一个实用的中间方案是：在Notebook中用`%run script.py`来调用已经模块化的代码，这样既保留了交互式探索的便利，又保持了代码的组织性。另一个好的实践是：把模型的不同变体定义为不同的函数（`build_model_v1()`、`build_model_v2()`等），而不是在同一块代码中注释掉旧版本、粘贴新版本。这样可以方便地对比不同架构的性能，也更容易回退到之前有效的配置。

#### 6.4.3 从手动反向传播到自动求导：何时用哪个

这一课我们重新使用了PyTorch的自动求导（autograd），而上一课我们花了大量时间手动实现反向传播。这引出了一个重要的问题：什么时候应该用自动求导，什么时候应该手动实现？

**使用自动求导的场景**：
- **架构探索阶段**：当你在设计新架构、调试超参数、尝试不同层组合时，自动求导让你可以专注于前向传播的设计，而不用担心反向传播的正确性。这大大提高了迭代速度。
- **复杂的网络结构**：当网络包含几十甚至上百层，涉及各种复杂的操作（如卷积、池化、注意力机制、跳跃连接）时，手动实现反向传播会变得极其繁琐且容易出错。
- **生产代码**：在部署到生产环境的代码中，自动求导是经过充分测试和优化的，可靠性远高于手动实现。

**手动实现反向传播的场景**：
- **学习和教学**：正如上一课所展示的，手动实现是建立深度理解的最佳方式。只有亲手推导过每个梯度，你才能真正理解网络在"学"什么。
- **调试梯度问题**：当模型不收敛或出现NaN时，手动检查特定层的梯度可以帮助你定位问题。有时候PyTorch的autograd可能由于某些操作产生意想不到的梯度，手动验证可以排除这种可能性。
- **自定义梯度**：当你需要实现一个PyTorch没有内置支持的操作，或者需要修改某个操作的梯度行为（如梯度裁剪、梯度惩罚）时，你必须手动控制反向传播。
- **极简部署**：在某些资源受限的环境中（如嵌入式设备），你可能无法使用完整的PyTorch，这时需要手动实现前向和反向传播。

在实际工作中，绝大多数时间你会使用自动求导。但上一课学到的技能会在关键时刻拯救你 —— 当你遇到奇怪的收敛问题时，能够手动检查梯度是否正确，这种能力是区分"调参工程师"和"真正理解深度学习的工程师"的重要标志。

#### 6.4.4 一个真实的调试故事

Karpathy在课中分享了一个真实的调试经历，深刻说明了深度学习开发的特性。当他在实现层次化BatchNorm时，最初的代码在2D输入（`(B, C)`）上工作正常，但当切换到3D输入（`(B, T, C)`）时，他忘记了调整均值和方差的计算维度。模型仍然能训练，loss也在下降，但最终验证loss比预期差了约0.01。这个差距很小，小到可能被当作随机波动忽略。但他坚持追踪，通过打印每一层的中间统计量，最终发现了`xmean`的形状不对 —— 应该是`(1, 1, C)`，但实际是`(1, T, C)`。

这个经历教会我们几件事：第一，深度学习中的bug很少是"爆炸性"的（如程序崩溃），更多是"静默的" —— 模型能跑，结果只是稍差一点。第二，跟踪中间张量的形状和统计量是发现这类bug的唯一方法。第三，即使是经验丰富的工程师（Karpathy曾是Tesla AI总监、OpenAI创始成员）也会犯这种看似"低级"的错误，因为张量维度的变化非常微妙。这完全正常，关键在于你有没有系统化的调试方法去发现并修复它们。

#### 6.4.5 训练过程监控：loss曲线与生成样本质量评估

训练完成后，评估模型需要特别注意一个关键细节：必须将BatchNorm切换到推理模式，使用训练期间累积的运行时统计量，而非当前batch的统计量。

```python
# 设置评估模式：对BatchNorm至关重要！
for layer in model.layers:
    layer.training = False

@torch.no_grad()
def split_loss(split):
    x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"{split:5s}: {loss.item():.4f}")

split_loss('train')  # 约 1.77
split_loss('val')    # 约 1.99
```

这里我们手动设置`layer.training = False`，因为我们使用的是自定义层而非PyTorch官方的`nn.Module`。在标准的PyTorch代码中，我们会调用`model.eval()`来统一设置所有子模块为评估模式，与之对应的是`model.train()`用于切换回训练模式。忘记在评估前调用`eval()`是深度学习开发中最常见的bug之一 —— 它不会导致程序报错，但会让评估loss异常偏高（因为BatchNorm使用当前小batch的统计量，可能非常不稳定），甚至让评估结果完全不可信。养成习惯：任何不涉及参数更新的代码（评估、测试、推理）之前都要调用`eval()`，涉及参数更新的训练循环之前都要调用`train()`。可以在代码中显式地加注释提醒自己，比如`# CRITICAL: set eval mode for BatchNorm`。

生成新名字的过程与之前相同，但现在模型可以利用8个字符的上下文来做出更informed的预测：

```python
for _ in range(20):
    out = []
    context = [0] * block_size  # 用'.'（索引0）初始化上下文窗口
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        # 从预测分布中采样（而非取argmax），保持生成多样性
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]  # 滑动窗口更新上下文
        out.append(ix)
        if ix == 0:  # 遇到结束符'.'
            break
    print(''.join(itos[i] for i in out))
```

随着上下文从3个字符增加到8个，模型生成的名字质量有可感知的提升。虽然验证集loss的下降幅度看似不大（从扁平结构的约2.027降到层次化结构的约1.993），但更长的上下文让模型能捕捉到更复杂的命名模式 —— 比如某些前缀（"sh"、"th"）对后续字符选择的约束，或者某些特定的三字母组合（如"-ing"、"-ton"）出现的概率。一个有趣的观察是：上下文从3增加到8带来的提升（约0.08的loss下降）比上下文从8进一步增加带来的提升要大得多，这暗示了对于英文名字这个特定任务，8个字符可能已经接近了"有用上下文"的上限。

下面是不同配置下的性能对比，从中我们可以看到一些有趣的规律：

| 配置 | 参数量 | 训练Loss | 验证Loss |
|---|---|---|---|
| 原始（3字符上下文 + 200隐藏层） | 12K | 2.058 | 2.105 |
| 上下文3→8（扁平展平） | 22K | 1.918 | 2.027 |
| 层次化树状结构（初始实现） | 22K | 1.941 | 2.029 |
| 修复BatchNorm维度bug后 | 22K | 1.912 | 2.022 |
| 扩大网络（n_embd=24, n_hidden=128） | 76K | **1.769** | **1.993** |

这个表格揭示了几个重要发现。首先，单纯增加上下文长度（从3到8）带来的改进是显著的，验证loss从2.105降到2.027 —— 这说明更多的上下文信息确实有帮助。其次，在参数量相同的情况下（22K），层次化结构的初始实现（训练loss 1.941）反而略差于扁平结构（1.918），这是因为存在一个微妙的BatchNorm维度处理bug。修复这个bug后，层次化结构以相同的参数量达到了更好的效果（训练loss 1.912 vs 1.918），这说明层次化结构本身是有优势的，但需要正确实现。这也提醒我们：一个"理论上更好"的架构如果实现有bug，可能还不如一个简单但正确实现的架构。最后，当我们扩大网络规模（n_embd=24, n_hidden=128）到76K参数时，验证loss进一步降到1.993 —— 这说明了在正确的架构下，适度增加参数量仍然是提升性能的有效手段。注意训练loss和验证loss之间的差距（约0.22）在所有配置中都相对稳定，说明模型并没有严重的过拟合问题，主要瓶颈在于模型容量或架构的表达能力。

回顾这一课的核心脉络：我们从扁平MLP的参数量爆炸问题出发，引入WaveNet的树状层次化结构来解决它。我们用`FlattenConsecutive`层实现了逐步聚合信息的机制，亲手搭建了一个完整的微型`torch.nn`框架来组织代码，学习了追踪张量形状和激活值分布等调试技能，最终理解了这种结构其实等价于一维卷积。这条从问题到解决方案到深入理解的脉络，正是Karpathy"从零到英雄"教学理念的精髓 —— 不是给你现成的答案，而是带你走过发现答案的完整过程，让你在每一步都理解"为什么"。

### 6.5 课后练习与推荐资源

#### 6.5.1 课后练习

1. **感受野的数学计算**：如果我们将`block_size`增加到16，需要多少层`FlattenConsecutive(2)`才能覆盖整个上下文窗口？如果改用`FlattenConsecutive(4)`（每次合并4个相邻位置），需要多少层？感受野的增长速度与每层的合并宽度有什么关系？请推导出一般公式：给定合并宽度$w$和层数$L$，总感受野为$w^L$。对于$block\_size=32$，分别计算$w=2$和$w=4$时所需的层数。如果每层$w=2$需要约50K参数，每层$w=4$需要约100K参数，哪种配置更参数高效？

2. **BatchNorm维度实验**：故意修改BatchNorm实现，让它在3D输入时只沿`dim=0`计算均值和方差（而非正确的`(0, 1)`）。训练模型并记录训练loss曲线和最终验证loss，与正确实现进行对比。思考：为什么这种"错误"的实现效果更差？打印中间变量`xmean`和`xvar`的形状来验证你的假设。从统计学的角度，对序列数据做归一化时，哪些维度应该被"展平"在一起？

3. **网络深度的系统实验**：固定`block_size=8`，尝试不同的网络深度配置。例如：只用2层FlattenConsecutive（此时最上层只能看到4个原始位置，感受野不够覆盖全部8个位置，观察这种"欠覆盖"对性能的影响）；或者使用4层配合更小的`n_embd`和`n_hidden`。记录每种配置下的参数量和验证loss，绘制"参数量 vs. 验证loss"的散点图。是否存在"过深"导致性能下降的情况？如果存在，为什么？

4. **手动反向传播挑战**：在不使用PyTorch自动求导的情况下，用上一课学到的手动反向传播知识，为WaveNet结构实现完整的反向传播。特别注意`view`操作的梯度处理 —— `view`在反向传播中对应的操作是另一个`view`，但形状要正确对应。三维张量通过各层Linear时的梯度维度也需要仔细追踪（提示：`Linear`对3D输入的处理是独立作用于最后一个维度的）。完成后，用上一课的`cmp`函数验证你的梯度与PyTorch autograd的结果一致。

5. **阅读PyTorch源码**：在GitHub上找到PyTorch的官方源码，阅读`nn.Module`、`nn.Sequential`、`nn.Linear`和`nn.BatchNorm1d`的实现。对比它们与本课中简化版本的异同。官方的`nn.Module`提供了哪些我们没实现的功能？比如`to(device)`（设备迁移，支持CPU/GPU切换）、`state_dict()`和`load_state_dict()`（模型保存加载，支持断点续训）、`apply()`（递归应用函数到所有子模块，常用于初始化）、`_modules`（有序的子模块字典，支持按名访问）、`zero_grad()`（统一清零梯度）、`named_parameters()`（带名称的参数迭代）等。理解这些功能是如何建立在相同的`__call__` + `parameters()`基础之上的。

6. **生成质量分析**：让训练好的模型生成1000个名字，统计它们与训练集中真实名字的相似度。计算有多少生成的名字恰好出现在训练集中（"记忆"），有多少是完全新颖的（"创造"）。分析不同架构（扁平vs层次化）和不同采样温度（temperature scaling）在生成多样性上的差异。温度参数可以通过在softmax之前将logits除以一个温度值$T$来实现：$p_i \propto \exp(z_i / T)$，其中$T > 1$会让分布更平缓（更多样化），$T < 1$会让分布更尖锐（更保守）。

7. **可视化层次化特征**：在模型的每一层聚合之后，提取中间表示`layer.out`，使用t-SNE或PCA将其降到2维并可视化。观察不同层级的表示是否确实捕捉了不同尺度的模式 —— 比如第1层的表示是否主要编码了相邻字符的关系，而第3层的表示是否编码了更长程的上下文信息。你可以将不同层的可视化结果并列比较，直观感受层次化抽象的演进过程。

#### 6.5.2 推荐学习资源

**官方资源**：
- [^2^] Karpathy课程视频：[Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- [^3^] GitHub Notebook：[makemore_part5_cnn1.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
- [^4^] makemore项目完整代码：[github.com/karpathy/makemore](https://github.com/karpathy/makemore)
- [^5^] 系列课程播放列表：[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

**WaveNet论文与扩展阅读**：
- [^1^] WaveNet原始论文：[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)（Aaron van den Oord et al., DeepMind, 2016）。这篇论文是深度生成模型的里程碑，其附录对dilated causal convolution的数学阐述非常清晰。即使你主要关注文本生成而非音频，论文中关于层次化感受野设计的洞察也极具启发性。特别关注论文中的Figure 3，它直观展示了dilation rate如何逐层增大感受野。
- BatchNorm原始论文：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)（Sergey Ioffe, Christian Szegedy, 2015）。深入理解BatchNorm对训练动态的影响，以及它在推理时为什么需要运行时统计量。

**PyTorch文档与教程**：
- [torch.nn.Module官方文档](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)：深入理解PyTorch模块系统的设计，特别是`__init__`、`forward`、`modules`、`parameters`等核心机制。
- [torch.nn.Sequential文档](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)：官方容器的使用方法，支持命名层和有序字典初始化。
- [PyTorch nn教程 — 从零开始构建](https://pytorch.org/tutorials/beginner/nn_tutorial.html)：PyTorch官方提供的替代教程，从另一个角度帮助你理解nn系统的设计思路。

**后续课程衔接**：在下一课中，我们将继续深入卷积神经网络，学习如何用PyTorch官方的`torch.nn.Conv1d`来替换本课中`FlattenConsecutive`加`Linear`的朴素实现，理解卷积核在序列上滑动共享权重的机制，探索padding、dilation、stride等超参数的作用，并进一步理解卷积在序列建模中的威力。这些知识将为之后学习GPT和Transformer架构打下坚实的基础。从MLP到CNN到Transformer，这是一个自然而然的演进脉络 —— 每种架构都在解决前一种的局限性，同时引入了新的设计思想。理解这个演进的历史脉络，比孤立地学习每种架构要有效得多。



---

## 7. 第7课：Let's build GPT — 从零构建Transformer

在前面的课程中，我们已经走过了很长一段路。从最简单的bigram模型，到带有隐层神经网络的字符级模型，再到引入网络深度和抽象层次的WaveNet风格层次化架构——我们一直在做同一件事：给定一段上下文序列，预测下一个字符。这个核心目标从未改变，但我们组织计算的方式却在不断进化。每一次架构上的改进，本质上都是在回答同一个问题：**如何让模型更有效地利用上下文信息？**

WaveNet通过空洞卷积（dilated Convolution）将感受野扩大到了数百个时间步，但它仍然有一个根本性的限制：序列中的每个位置都必须经过层层卷积的逐次传播才能"传播"到远处。信息在长距离流动时，会穿过许多中间层，每层都可能带来信息的损失或扭曲。如果我们能让序列中任意两个位置直接对话——不需要经过中间步骤——会怎样？这就是Transformer[^1^]提出的革命性思想，也是本章要实现的完整架构。

### 7.1 GPT架构概览

#### 7.1.1 GPT = Generative Pretrained Transformer：自回归语言模型

**GPT** 的全称是 **Generative Pretrained Transformer**，直译为"生成式预训练Transformer"。这个名字拆解开来，恰好描述了它的三个核心特征：

**Generative（生成式）**：模型被训练来生成序列数据。给定一个前缀（prompt），它会逐字符（或逐token）地生成后续内容。这与判别式模型（如分类器）形成对比——GPT不做"是/否"的判断，而是做"接下来该写什么"的创作。

**Pretrained（预训练）**：模型先在一个大规模无标注语料上进行通用语言建模训练，学习语言的统计规律和结构。预训练完成后，模型可以通过微调（fine-tuning）适应特定下游任务。GPT-1、GPT-2、GPT-3都遵循这个范式，只是规模越来越大。

**Transformer**：这是模型的架构基础。GPT使用的是Transformer架构的**decoder-only**变体——只保留解码器部分，移除了编码器-解码器交叉注意力。这种设计天然适合自回归生成任务。

GPT的训练目标极其简单：给定序列 $x_1, x_2, \ldots, x_T$，最大化条件概率的连乘积：

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

也就是说，每个位置 $t$ 的token，其预测只能依赖于位置 $1$ 到 $t-1$ 的信息。这种"只看过去，不看未来"的约束称为**自回归性质（autoregressive property）**，它是GPT能够逐token生成文本的基础。在训练时，我们通过**teacher forcing**技术一次性计算所有位置的损失：对序列中每个位置 $t$，都用前 $t-1$ 个token作为输入来预测第 $t$ 个token，然后对所有位置的预测误差求平均。

#### 7.1.2 与之前makemore的关系：相同的语言建模目标，不同的架构

如果你从第2课的makemore系列一路学习过来，会发现一个令人欣慰的事实：**目标函数从未改变**。无论是简单的bigram查找表、单隐层神经网络、还是WaveNet风格的卷积架构，我们始终在做同一件事——最大化训练数据的对数似然，即最小化交叉熵损失。变化的只是 $P(x_t \mid x_{<t})$ 这个条件分布的建模方式。

| 架构 | 上下文利用方式 | 长距离依赖的处理 | 主要限制 |
|------|--------------|----------------|---------|
| Bigram | 仅前一个token | 无 | 无法利用任何上下文 |
| MLP | 固定窗口的token拼接 | 受限于窗口大小 | 上下文长度受限 |
| WaveNet | 空洞卷积逐层传播 | 感受野随深度指数增长 | 信息需层层传播，非直接连接 |
| **Transformer** | **Self-Attention直接全连接** | **任意距离一步可达** | **计算复杂度为$O(T^2)$** |

Transformer用Self-Attention机制替代了序列化的信息传播。在Attention层中，序列中的每个位置都可以**直接** attending 到所有其他位置——不需要经过中间层的传递。这从根本上解决了长距离依赖问题：无论两个token在序列中相距多远，它们之间的交互路径长度始终为1。

#### 7.1.3 Transformer的核心创新：Self-Attention机制替代序列计算

在Transformer出现之前，深度序列模型几乎都被循环架构（RNN、LSTM、GRU）或卷积架构（WaveNet、ByteNet）主导。这两类架构的共同点是**局部性和序列性**：信息必须通过相邻位置逐步传播，远距离的信息交互需要经过许多中间步骤。

Self-Attention（自注意力机制）打破了这个范式。它的核心思想可以用一句话概括：**每个位置都计算自己与所有其他位置的"相关性"，然后用这些相关性作为权重，对所有位置的信息做加权聚合**。这样一来，序列中任意两个位置之间的信息流动都是直接的、一步到位的。

这种设计的优雅之处在于：它保留了序列的全局依赖能力，同时可以高度并行化。在一个Attention层内，所有位置之间的交互计算可以同时进行——这与RNN的严格串行计算形成鲜明对比。

#### 7.1.4 "Attention is All You Need"论文(2017)的核心贡献

Transformer架构诞生于2017年Google发表的论文"Attention is All You Need"[^1^]。这篇论文的标题本身就是一个宣言：作者们证明，**完全不需要循环或卷积，仅依靠注意力机制就能构建出强大的序列转导模型**。

论文提出了两种架构变体：

- **Encoder-Decoder Transformer**：用于机器翻译等seq2seq任务。编码器处理输入序列，解码器生成输出序列，两者之间通过交叉注意力（cross-attention）连接。
- **Decoder-only Transformer**：仅使用解码器部分，配合causal masking进行自回归语言建模。这就是GPT系列采用的架构。

Karpathy在本课中构建的是decoder-only GPT。我们不会实现完整的编码器-解码器结构——因为对于自回归文本生成任务，decoder-only已经足够了。这种简化后来成为大型语言模型（LLM）的主流设计选择，GPT-2[^2^]、GPT-3[^3^]、ChatGPT，以及后来的Llama、Claude等模型都遵循这一范式。

### 7.2 Self-Attention机制

Self-Attention是整个Transformer架构中最核心的模块，也是理解GPT工作原理的关键。让我们从问题出发，一步步推导出它的数学形式和代码实现。

#### 7.2.1 问题定义：每个位置需要聚合所有先前位置的信息

想象你正在读一句话："The cat sat on the ___ and looked at the mouse." 要预测空白处的词，你需要综合整句话的信息——"cat"暗示了可能是一个与猫相关的东西，"sat on"暗示了某种平面物体，"mouse"则进一步确认了场景。所有这些线索来自句子中不同的位置。

在神经网络中，每个位置 $t$ 需要一个**表示（representation）**，这个表示应该融合来自所有相关位置的信息。Self-Attention要回答的问题是：**如何用一种可学习的方式，让每个位置"挑选"出它应该关注哪些其他位置，并从这些位置获取信息？**

Karpathy在课程中用了一个非常直观的渐进式讲解。他首先展示了最naive的聚合方式——用for循环做简单平均：

```python
# version 1: averaging past context with for loops (the weakest form of aggregation)
# x[b,t] = mean_{i<=t} x[b,i]
B, T, C = x.shape
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]           # (t, C) — 取位置0到t的所有向量
        xbow[b,t] = torch.mean(xprev, 0)  # 对时间维度求平均
```

这段代码实现了最基础的"上下文聚合"：每个位置 $t$ 的表示，是位置 $0$ 到 $t$ 所有向量的算术平均。这种方式虽然能让信息从前面流过来，但存在严重问题：**每个前面位置的权重都相等**。无论某个词与当前词是否相关，它都贡献相同的平均权重。这显然不是理想的做法。

我们需要一种机制，让每个位置根据内容动态地决定"应该关注谁，关注多少"。这就是Self-Attention的核心思想。

#### 7.2.2 注意力权重：查询(query)与键(key)的点积衡量位置间相关性

Self-Attention的精妙之处在于它引入了三个可学习的线性投影，将每个输入向量映射为三种不同的角色：

- **Query (Q, 查询)**：当前位置"想要获取什么信息"的表示。每个位置发出一个query，询问"哪些位置的信息与我最相关？"
- **Key (K, 键)**：每个位置"能提供什么信息"的表示。key相当于每个位置的"标签"或"标识"，用于回答query的询问。
- **Value (V, 值)**：每个位置实际携带的信息内容。当位置 $i$ 被选中时，我们真正取走的就是它的value。

这三个投影都是简单的线性层（`nn.Linear`），没有偏置项（`bias=False`）。它们的权重矩阵是在训练中学习的——模型通过梯度下降自动学会什么样的query-key匹配意味着"这两个位置相关"。

注意力分数的计算基于query和key的点积：

$$\text{score}(q_t, k_s) = q_t \cdot k_s$$

如果query和key方向相近（点积大），意味着这两个位置在语义上相关；如果方向正交或相反（点积小或为负），意味着不相关。为了让数值在训练初期保持稳定，我们还会除以 $\sqrt{d_k}$（$d_k$ 是query/key的维度），这个缩放因子称为**scaled dot-product attention**。

为什么除以 $\sqrt{d_k}$？考虑一个直观的场景：当 $d_k$ 很大时，两个随机向量在高维空间中的点积绝对值会自然增大（维度越高，随机分量累加越多）。如果不做缩放，softmax的输入会非常大，导致梯度变得极其稀疏，训练不稳定。除以 $\sqrt{d_k}$ 可以将点积的方差大致归一化到1，保持softmax输入在一个合理的范围内。

#### 7.2.3 加权聚合：用softmax归一化的注意力权重对值(value)做加权平均

得到了query和key之间的点积分数后，下一步是将它们转化为**概率分布**（即注意力权重），然后用这些权重对value做加权求和。

Softmax的作用是将任意实数向量转化为非负且和为1的概率分布。在注意力机制中，这意味着每个位置会分配一部分"注意力预算"给所有其他位置——预算总和为1，重要的位置获得更多，不重要的位置获得更少。

加权聚合的完整计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

这个公式的形状变化值得仔细跟踪。假设输入 $X$ 的形状为 $(B, T, C)$，其中 $B$ 是batch size，$T$ 是序列长度，$C$ 是embedding维度。通过线性投影后：

- $Q = XW_Q$ 的形状为 $(B, T, d_k)$
- $K = XW_K$ 的形状为 $(B, T, d_k)$
- $V = XW_V$ 的形状为 $(B, T, d_v)$

然后 $QK^T$ 的形状是 $(B, T, T)$ —— 这是一个**注意力矩阵**，每一行代表一个query对所有key的分数。经过softmax后，每行和为1，形成合法的注意力权重分布。最后乘以 $V$，得到输出 $(B, T, d_v)$ —— 每个位置的输出是**所有位置value的加权平均**，权重由query-key匹配度决定。

#### 7.2.4 Causal Masking：确保位置i只能看到位置≤i的信息，保持自回归性质

现在有一个关键问题：在语言建模任务中，位置 $t$ 在预测下一个token时，**不能看到位置 $t$ 之后的未来信息**。如果模型能看到未来的答案，它就学会了"作弊"而不是真正理解语言规律。

为了确保这一点，GPT使用**Causal Masking**（因果掩码，也叫自回归掩码或下三角掩码）。具体来说，我们将注意力矩阵的上三角部分（即"未来"位置对应的分数）设为 $-\infty$。经过softmax后，这些位置的权重变为0——相当于完全屏蔽了未来的信息。

实现上，这个mask是一个下三角矩阵，通过 `torch.tril` 创建：

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
```

`register_buffer` 告诉PyTorch这是一个不参与梯度计算的持久化张量。`torch.tril` 生成一个下三角矩阵：对角线及以下位置为1，以上位置为0。

在forward中应用mask：

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

`masked_fill` 将所有mask为0的位置（即上三角部分）填充为 `-inf`。Softmax遇到 `-inf` 时会输出0，完美实现了对未来信息的屏蔽。

值得注意的是，**这个mask在训练期间是固定的**，不是学习得到的。它是我们对模型结构的先验约束——我们硬编码了"不允许偷看未来"的规则。这也是GPT被称为decoder-only架构的原因：编码器-解码器Transformer中的编码器没有这种单向mask，它可以让每个位置看到整个输入序列的左右上下文；而GPT的decoder必须保持自回归性质。

#### 7.2.5 完整Self-Attention Head的代码实现

现在让我们把上面的所有概念整合为一个完整的Self-Attention Head实现。这是本课最核心的代码，每一个模块都建立在此基础上。

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)     # (B,T,hs) — key vectors
        q = self.query(x)   # (B,T,hs) — query vectors
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)  # (B,T,T) — attention weights sum to 1
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs) — value vectors
        out = wei @ v      # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
```

让我们逐行解析这个实现：

**初始化部分**：三个`nn.Linear`层分别对应key、query、value的线性投影。`bias=False`是因为注意力分数的归一化由softmax完成，偏置项不是必需的。`register_buffer`注册了causal mask，它不参与梯度计算但在每次forward时都会被使用。`dropout`在注意力权重上随机置零一部分连接，作为正则化手段。

**Forward计算**：
1. `k = self.key(x)` 和 `q = self.query(x)`：通过线性投影得到key和query向量，形状为 $(B, T, \text{head\_size})$。
2. `wei = q @ k.transpose(-2, -1)`：计算query-key点积，得到注意力分数矩阵，形状为 $(B, T, T)$。`transpose(-2, -1)`交换最后两个维度，使得矩阵乘法结果为(batch, T, T)。
3. `* k.shape[-1]**-0.5`：这是scaled dot-product的缩放因子，等价于除以 $\sqrt{d_k}$。
4. `masked_fill`：应用causal mask，将未来位置设为 `-inf`。
5. `F.softmax(wei, dim=-1)`：在最后一个维度（即每行内）做softmax归一化，得到注意力权重。
6. `self.dropout(wei)`：训练时对注意力权重做随机dropout。
7. `v = self.value(x)`：通过线性投影得到value向量。
8. `out = wei @ v`：用注意力权重对value做加权求和，输出形状为 $(B, T, \text{head\_size})$。

这个实现虽然简洁，却完整地表达了Scaled Dot-Product Self-Attention的所有核心步骤。理解了这个Head模块，你就理解了Transformer 80%的工作原理。

### 7.3 Multi-Head Attention

#### 7.3.1 多头注意力的直觉：多组Q/K/V并行计算，捕捉不同类型的依赖关系

单个Self-Attention Head虽然已经很强大，但它有一个局限：**所有位置之间的交互都被压缩到一组Q/K/V投影中**。这意味着一个head只能学习一种类型的关注模式。但在自然语言中，token之间的关系是多方面的：一个词可能与前面的词有主谓关系，与远处的词有指代关系，与相邻的词有修饰关系——不同类型的依赖关系需要不同的方式来捕捉。

**Multi-Head Attention（多头注意力）** 的直觉非常直接：与其只用一组Q/K/V，不如并行使用多组。每个head有自己的一套投影矩阵，独立计算注意力。不同head可以专门化地学习不同类型的依赖关系——一个head可能关注局部语法模式，另一个head可能捕捉长距离语义关联。

类比来说，单头注意力就像一个人用一只眼观察世界；多头注意力就像同时用多只眼观察，每只眼对不同类型的特征敏感。最终我们将所有"眼"的观察结果合并，得到更丰富的表示。

#### 7.3.2 实现方式：将embedding分成多组，分别计算注意力

Multi-Head Attention的实现非常直接：我们创建 `num_heads` 个独立的 `Head` 实例，在forward中将输入同时传递给所有head，然后将它们的输出拼接在一起。

```python
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

这里的关键是 `head_size` 的选择。每个head的输出维度是 `head_size`，`num_heads` 个head拼接后的总维度是 `head_size * num_heads`。在Karpathy的实现中，`head_size = n_embd // n_head`，这意味着拼接后的维度恰好等于原始的embedding维度 `n_embd`。例如，如果 `n_embd = 384`，`n_head = 6`，则每个head的维度是64，6个head拼接后就是 $6 \times 64 = 384$。

这种设计保持了维度的一致性：MultiHeadAttention的输入和输出都是 `n_embd` 维，这使得它可以无缝地嵌入到残差连接中（后续我们会看到）。

#### 7.3.3 拼接与投影：将多头的输出拼接后做一次线性变换

`torch.cat([h(x) for h in self.heads], dim=-1)` 将所有head的输出在最后一个维度（通道维度）上拼接。假设每个head输出 $(B, T, 64)$，6个head拼接后得到 $(B, T, 384)$。

但简单拼接有一个问题：不同head的输出可能只是机械地堆叠，没有真正的信息融合。为了解决这个问题，我们在拼接后添加了一个投影层 `self.proj = nn.Linear(head_size * num_heads, n_embd)`。这个线性变换让不同head的信息可以互相混合——相当于在所有head的结果之上再做一次特征重组。

`self.dropout(self.proj(out))` 在投影之后应用dropout，进一步防止过拟合。这种"多头计算 + 投影融合"的设计是Transformer的标准做法，原始论文[^1^]也是这样描述的。

值得一提的是，Karpathy在课程中还提到了一个高级优化：**将多个head的Q/K/V计算合并为一个大的矩阵乘法**。在原始论文和高效实现（如nanoGPT）中，不会为每个head单独创建`nn.Linear`层，而是创建一个输出维度为 `n_embd` 的大线性层，然后在计算中将结果reshape为 `(B, T, n_head, head_size)`，把head维度视为batch维度统一处理。这种方式更高效，因为它将多个小矩阵乘法合并为一个大的、硬件更友好的矩阵乘法。Karpathy将这个作为课后练习留给了学习者。

### 7.4 Transformer Block的完整组件

Multi-Head Attention提供了token之间的"通信"机制，但一个完整的Transformer Block还需要其他几个关键组件。让我们逐一分析。

#### 7.4.1 Feed-Forward层：每个位置独立的两层MLP，扩展再压缩维度

Attention层负责的是token**之间**的信息交流——它让每个位置能够聚合其他位置的信息。但聚合之后，每个位置还需要对聚合后的信息进行**非线性变换**，以学习更复杂的特征表示。这就是**Feed-Forward Network（FFN，前馈神经网络）**的作用。

```python
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

FFN的结构非常简单：先通过一个线性层将维度从 `n_embd` 扩展到 `4 * n_embd`，经过ReLU非线性激活，再通过另一个线性层压缩回 `n_embd`。为什么中间要扩展4倍？原始论文[^1^]就是这么设定的，后续工作发现这个比例在计算效率和模型容量之间取得了不错的平衡。扩展维度相当于在更宽的特征空间中做非线性变换，这增加了模型的表达能力。

注意FFN的 crucial 特性：**它对序列中的每个位置独立操作**。FFN没有位置间的交互——它只是逐位置地将 $n\_embd$ 维向量映射到另一个 $n\_embd$ 维向量。位置之间的交互只在Attention层发生。这种分工非常清晰：Attention负责"通信"（communication），FFN负责"计算"（computation）。

#### 7.4.2 Residual Connections：绕过大块计算，缓解梯度消失，帮助深层训练

当我们堆叠多个Transformer Block时，会面临一个经典问题：**梯度消失**。在反向传播时，梯度需要穿过一层又一层的非线性变换，每穿过一层都会被缩放。如果网络很深，梯度在到达浅层时可能已经变得极小，导致这些层的参数几乎不更新。

**Residual Connection（残差连接，也叫跳跃连接/skip connection）** 是解决这个问题的方法。它最早在ResNet[^4^]中提出，核心思想极其简单：将子层的输入直接加到子层的输出上。

$$\text{output} = x + \text{Sublayer}(x)$$

在反向传播时，残差连接提供了一条"高速公路"：梯度可以直接通过 `+` 操作的加法分支反向传播，而不需要穿过Sublayer的复杂计算。这意味着即使深层网络，浅层的参数也能接收到足够强的梯度信号。

残差连接的另一个好处是它让网络可以**逐步学习**。在训练初期，子层可能还没有学到有用的变换，输出接近于0或随机值。此时残差连接让信息可以几乎无损地流过（因为 $x + \text{small} \approx x$），相当于给网络一个"保底"的身份映射。随着训练进行，子层逐渐学会在身份映射的基础上做增量改进。

#### 7.4.3 Layer Normalization：在每个子层前做归一化(Pre-Norm架构)

深度网络训练中的一个核心挑战是**内部协变量偏移（internal covariate shift）**：每一层的输入分布在训练过程中不断变化，因为前面层的参数在更新。这迫使每一层必须不断适应变化的输入分布， slows down training。

**Layer Normalization（层归一化）** 对每个样本的特征维度进行归一化，使得输出的均值为0、标准差为1：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu$ 和 $\sigma^2$ 是对当前样本的所有特征维度计算的均值和方差，$\gamma$ 和 $\beta$ 是可学习的缩放和平移参数，$\epsilon$ 是一个小常数防止除零。

LayerNorm与BatchNorm的关键区别在于：LayerNorm对每个样本独立归一化（在特征维度上做统计），而BatchNorm对一个batch的样本做归一化（在batch维度上做统计）。对于序列建模任务，序列长度可能变化，而且我们希望在推理时行为确定——LayerNorm的这些特性使其更适合Transformer。

Karpathy的代码使用了一种现代Transformer训练更稳定的变体：**Pre-LayerNorm**（在子层输入前做归一化）。原始"Attention is All You Need"论文使用的是Post-LayerNorm（在子层输出后做归一化），但后来研究发现Pre-Norm让深层Transformer的训练更加稳定。Pre-Norm的结构是：

```python
x = x + Sublayer(LayerNorm(x))   # Pre-Norm
```

而不是原始的：

```python
x = LayerNorm(x + Sublayer(x))   # Post-Norm
```

这个差异看似微小，但对训练的稳定性有显著影响。在Pre-Norm中，LayerNorm的输入来自上一层的残差路径，分布更加稳定；同时残差连接 "绕过" 的是归一化后的子层输出，梯度流动更加顺畅。

#### 7.4.4 Block的组装：Attention → Add&Norm → FFN → Add&Norm

现在我们可以将所有组件组装为一个完整的Transformer Block：

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # communication with pre-norm residual
        x = x + self.ffwd(self.ln2(x))  # computation with pre-norm residual
        return x
```

一个Block包含两个子层，每个子层都遵循"Pre-Norm + 子层计算 + 残差连接"的模式：

1. **第一子层（通信层）**：`x = x + self.sa(self.ln1(x))`。输入先经过LayerNorm，然后通过Multi-Head Attention做token间的信息聚合，最后与原始输入做残差相加。这实现了"每个位置收集其他位置的信息"的功能。

2. **第二子层（计算层）**：`x = x + self.ffwd(self.ln2(x))`。输入再次经过LayerNorm，然后通过Feed-Forward Network做逐位置的非线性变换，再与输入做残差相加。这实现了"每个位置对收集到的信息做深度处理"的功能。

这种"通信+计算"的组合是Transformer Block的标准设计。当我们堆叠多个Block时，信息可以在序列位置间多次流动（通过Attention层），同时在每个位置上也经过多次非线性变换（通过FFN层）。深层Transformer通过这种交替的通信和计算，逐步构建出越来越抽象的表示。

### 7.5 完整GPT模型

#### 7.5.1 Token Embedding + Positional Embedding：给模型位置信息

到目前为止，我们讨论的Self-Attention有一个重要的性质：**它是置换等变的（permutation-equivariant）**。也就是说，如果你打乱输入序列的顺序，注意力输出的对应位置也会被同样打乱——Attention本身不"知道"哪个token在序列的哪个位置。它只关心token之间的内容关系，不关心空间/时间关系。

但在语言中，**顺序是至关重要的**。"猫追老鼠"和"老鼠追猫"是完全不同的意思，包含完全相同的token，只是顺序不同。为了让模型感知位置信息，Transformer引入了**Positional Embedding（位置嵌入）**。

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

模型的第一层包含两个embedding查找表：

- **Token Embedding**：将每个token ID映射为一个 `n_embd` 维的稠密向量。这相当于一个可学习的"词向量"查找表，每个字符（或token）都有一个唯一的向量表示。在训练过程中，这些向量会随着梯度下降不断调整，使得语义相近的字符在向量空间中靠近。

- **Position Embedding**：将每个位置索引（0, 1, 2, ..., block_size-1）映射为一个 `n_embd` 维的向量。这意味着模型为序列中的每个"座位号"学习了一个特定的向量。无论坐在第3个位置上的是哪个token，它都会获得相同的位置向量。

在forward中，两个embedding被相加：

```python
tok_emb = self.token_embedding_table(idx)           # (B,T,C)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
x = tok_emb + pos_emb  # (B,T,C)
```

注意位置embedding的巧妙索引：`torch.arange(T)` 生成 `[0, 1, ..., T-1]`，这意味着位置0的token获得position embedding的第0行，位置1的token获得第1行，以此类推。由于PyTorch的广播机制，形状为 $(T, C)$ 的 `pos_emb` 会自动广播到 $(B, T, C)$，与 `tok_emb` 相加。

这种可学习的位置嵌入是原始Transformer论文中的方案。后来也出现了其他变体，如正弦位置编码（sinusoidal positional encoding，不需要学习参数）和旋转位置编码（RoPE，目前大模型中最流行的方案）。Karpathy使用的是最简单的可学习嵌入，因为它简单有效。

#### 7.5.2 N个Transformer Block的堆叠：从GPT-2 small到GPT-3的尺度差异

一个Block是一个"通信+计算"的单元。GPT的强大之处在于**堆叠多个这样的Block**，让每个token的信息可以在序列中深度传播和反复处理。

```python
self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
```

这里 `n_layer = 6`，意味着我们堆叠6个Transformer Block。数据流过这些Block时，每个Block的Attention层让信息在token间流动，FFN层在每个位置做非线性变换。经过6轮这样的处理，模型能够捕捉到相当复杂的语言模式。

值得注意的是，从GPT-2到GPT-3，模型的核心架构几乎没变——变化的主要是**规模**。下表对比了几个代表性配置的参数量：

| 配置 | n_layer | n_embd | n_head | 参数量 |
|------|---------|--------|--------|--------|
| 本课实现 | 6 | 384 | 6 | ~1000万 |
| GPT-2 small | 12 | 768 | 12 | 1.24亿 |
| GPT-2 medium | 24 | 1024 | 16 | 3.55亿 |
| GPT-2 large | 36 | 1280 | 20 | 7.74亿 |
| GPT-2 XL | 48 | 1600 | 25 | 15.5亿 |
| GPT-3 | 96 | 12288 | 96 | 1750亿 |

可以看到，GPT-3[^3^]的embedding维度是12288，层数是96，总参数量达到了1750亿。但核心组件——多头自注意力、前馈网络、残差连接、层归一化——与我们在本课中实现的是完全相同的。这就是Transformer架构的可扩展性之美：**你不需要改变架构，只需要把它做得更大**。

#### 7.5.3 输出头：最终Linear层将表示映射到vocab上的概率分布

经过所有Transformer Block的处理后，每个位置都有一个 `n_embd` 维的向量表示。为了进行语言建模（预测下一个token），我们需要将这些向量映射到词汇表上的概率分布。这是通过一个最终的线性层实现的：

```python
self.lm_head = nn.Linear(n_embd, vocab_size)
```

`lm_head`（language model head）将每个位置的 `n_embd` 维表示映射为 `vocab_size` 维的logits向量。在训练时，这些logits与目标token计算交叉熵损失：

```python
logits = self.lm_head(x)  # (B, T, vocab_size)
B, T, C = logits.shape
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```

这里`view`操作将三维张量reshape为二维，因为`F.cross_entropy`期望输入形状为`(N, C)`，其中`N`是样本数，`C`是类别数。我们将`B*T`个位置独立地计算交叉熵，PyTorch会在内部自动取平均。

在生成时，我们只关心序列最后一个位置的预测（因为自回归生成中，下一个token只取决于已经生成的所有token）：

```python
logits = logits[:, -1, :]  # 只取最后一个时间步 (B, vocab_size)
probs = F.softmax(logits, dim=-1)  # 转为概率分布
idx_next = torch.multinomial(probs, num_samples=1)  # 采样
```

`torch.multinomial` 根据概率分布采样下一个token。`softmax` 的温度默认是1——如果你希望生成更"确定"（更保守）或更"随机"（更有创造性）的文本，可以通过调整logits的温度来控制。温度是文本生成中一个重要的超参数。

#### 7.5.4 完整模型代码与参数计数

现在让我们把所有组件整合为完整的GPT语言模型：

```python
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init — not covered in the original GPT video, but important
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)                         # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)     # (B,T,C)
        x = self.ln_f(x)       # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
```

这个实现包含了几个值得注意的细节：

**权重初始化**：`_init_weights` 方法对Linear和Embedding层使用了均值为0、标准差为0.02的正态分布初始化。良好的初始化对Transformer训练至关重要——如果权重初始值太大，softmax会在训练初期就饱和；如果太小，信号无法有效传播。标准差0.02是一个经验值，在GPT-2中也使用了类似的设置。

**最终LayerNorm**：在所有Block之后，我们额外添加了一个 `ln_f`（final layer norm）。这确保了进入输出头之前，表示的分布已经被归一化。Pre-Norm架构中，每个子层前都有LayerNorm，但最后还需要一个来保证输出头的输入稳定。

**生成时的上下文裁剪**：`idx_cond = idx[:, -block_size:]` 这一行很关键。由于位置embedding只有 `block_size` 个条目，模型无法处理超过这个长度的序列。在生成过程中，随着序列不断增长，我们必须只保留最后 `block_size` 个token作为上下文。这意味着模型在长文本生成时，可能会"遗忘"太远之前的内容——这是所有固定上下文长度模型的固有限制。

模型创建后，我们可以统计参数量：

```python
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
```

在本课的配置下（vocab_size=65, n_embd=384, n_head=6, n_layer=6），这个模型大约有**1000万**参数。虽然远小于GPT-2的1.24亿，但已经足够在shakespeare数据集上生成令人印象深刻的莎士比亚风格文本。

### 7.6 训练与生成

#### 7.6.1 数据准备：加载shakespeare数据集，构建batch

我们继续使用tinyshakespeare数据集——William Shakespeare的全部作品约100万字符。字符级tokenization非常简单：每个唯一字符分配一个整数ID。

```python
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]            # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l])   # decoder: list of integers -> string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))    # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

`get_batch` 函数随机从训练或验证集中抽取 `batch_size` 个长度为 `block_size` 的序列。注意目标序列 `y` 是输入序列 `x` 向左平移一位的结果：如果 $x = [x_0, x_1, x_2, \ldots, x_{T-1}]$，那么 $y = [x_1, x_2, x_3, \ldots, x_T]$。这意味着模型在位置 $t$ 的目标输出就是 $x_{t+1}$ —— 下一个token。

`torch.randint` 生成随机起始索引，确保我们不会越界（最后一个有效的起始索引是 `len(data) - block_size - 1`）。`torch.stack` 将 `batch_size` 个一维张量堆叠为一个二维张量，形状为 `(batch_size, block_size)`。

#### 7.6.2 训练循环：与之前相同的模式，但规模更大

训练循环的模式与makemore系列完全一致——这不是巧合，因为所有自回归语言模型共享相同的训练范式：

```python
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

我们使用 **AdamW** 优化器，它是Adam的一种变体，对权重衰减（weight decay）的处理更正确。学习率设为 `3e-4`，这是Transformer训练中的常见默认值。

`estimate_loss` 函数在训练集和验证集上分别评估多轮（`eval_iters=200`轮），取平均损失。这给了我们更稳定的损失估计，避免因batch的随机性导致的波动。如果训练损失持续下降但验证损失开始上升，说明模型开始过拟合。

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

注意 `@torch.no_grad()` 装饰器：在评估时我们不需要计算梯度，这节省了大量内存和计算。`model.eval()` 将模型设为评估模式（影响dropout和batchnorm等行为），评估完成后用 `model.train()` 恢复训练模式。

#### 7.6.3 文本生成：从提示开始，反复预测下一个token并拼接

训练完成后，我们可以通过 `generate` 方法让模型创作文本：

```python
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

`context = torch.zeros((1, 1))` 创建了一个以token ID 0（在本课的字符级编码中对应换行符 `\n`）为起始的prompt。`generate` 方法会重复500次：预测下一个token、将其拼接到序列末尾、用更新后的序列再次预测。

如果你给模型一个更有意义的起始prompt，比如 `"ROMEO: "`，它会尝试以莎士比亚风格续写：

```python
context = torch.tensor(encode("ROMEO: "), dtype=torch.long, device=device).unsqueeze(0)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

生成过程本质上是**从条件分布中逐token采样**。由于我们使用 `torch.multinomial`（多项式采样）而非 `torch.argmax`（贪婪解码），每次生成的结果都会有所不同。这种随机性让生成文本更加多样化。如果你想减少随机性、让输出更加"确定"，可以在softmax前对logits除以一个温度参数 $T < 1$（使分布更尖锐），或者在采样时使用top-k或nucleus sampling等高级技术。

#### 7.6.4 GitHub Copilot辅助写GPT的元体验

Karpathy在课程末尾分享了一个有趣的观察：他当时正在使用GitHub Copilot（一个基于Codex模型的AI编程助手）来编写GPT的代码。Copilot本身就是一个GPT模型，它在海量代码上训练过，能根据上下文自动补全代码。

这创造了一个有趣的"元体验"：**用GPT来写GPT**。当你开始输入`class Head(nn.Module):`时，Copilot可能已经猜到了你接下来要写Self-Attention的逻辑，并自动补全了query、key、value的投影和注意力计算。这不仅提高了编码效率，也从侧面证明了Transformer架构的"模式识别"能力——GPT模型学会了理解GPT架构的代码模式。

这个观察揭示了一个深层道理：大型语言模型不仅能生成自然语言文本，还能生成代码、数学推导、甚至对自身架构的描述。这种通用性正是Transformer架构和自回归预训练范式强大之处的体现。

### 7.7 课后练习与资源

#### 7.7.1 练习题

**练习1：高效Multi-Head Attention实现**

当前`MultiHeadAttention`的实现为每个head单独创建了一个`Head`实例，这会导致多个小型矩阵乘法。在高效实现中，通常将head维度视为batch维度，用一个大的线性层同时计算所有head的Q/K/V。尝试将`Head`和`MultiHeadAttention`合并为一个类，通过将`n_head`视为另一个batch维度来并行处理所有head。提示：将线性层的输出reshape为`(B, T, n_head, head_size)`，然后transpose为`(B, n_head, T, head_size)`，这样注意力计算可以在所有head上并行进行。参考实现可以在nanoGPT仓库中找到。

**练习2：训练一个数学GPT**

用你自己的数据集训练GPT。一个有趣的任务是训练GPT做两个数字的加法（例如输入"123+456="，期望输出"579"）。提示：你可能需要将输出数字序列反转（因为低位数字在加法中先被计算），并使用`CrossEntropyLoss`的`ignore_index`参数来掩码输入位置的损失——我们只对输出部分的预测计算损失。

**练习3：预训练+微调**

找到一个非常大的数据集（确保训练损失和验证损失没有明显差距，说明数据充足），先在该数据上预训练Transformer，然后在tinyshakespeare上以更少的步数和更低的学习率微调。观察是否能获得比直接训练更低的验证损失。这个练习模拟了现代LLM开发中的"预训练+微调"范式。

**练习4：条件生成**

修改模型使其支持条件生成——给定一个特定的风格或主题提示（如"喜剧："或"悲剧："），生成对应风格的文本。你可以通过在训练数据前添加风格前缀来实现。

#### 7.7.2 推荐学习资源

**核心论文**：

- "Attention is All You Need"（Vaswani et al., 2017）[^1^] — Transformer的原始论文。建议仔细阅读第3节的模型架构描述和第4节的注意力机制分析。论文地址：https://arxiv.org/abs/1706.03762

- "Language Models are Unsupervised Multitask Learners"（Radford et al., 2019）[^2^] — GPT-2论文。展示了在WebText数据集上大规模预训练的语言模型如何在多个下游任务上zero-shot表现出色。论文地址：https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

- "Language Models are Few-Shot Learners"（Brown et al., 2020）[^3^] — GPT-3论文。证明了175B参数的GPT-3可以通过少量示例（few-shot prompting）完成多种NLP任务，无需微调。论文地址：https://arxiv.org/abs/2005.14165

**Karpathy官方资源**：

- 本课视频（720万+观看）: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Google Colab配套Notebook: https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L
- ng-video-lecture仓库（完整代码）: https://github.com/karpathy/ng-video-lecture
- nanoGPT仓库（生产级GPT训练）: https://github.com/karpathy/nanoGPT

**扩展阅读**：

- "Deep Residual Learning for Image Recognition"（He et al., 2016）[^4^] — ResNet论文，提出了残差连接。虽然针对图像任务，但残差连接的思想对所有深度网络都适用。论文地址：https://arxiv.org/abs/1512.03385

- "Layer Normalization"（Ba et al., 2016）[^5^] — Layer Normalization的原始论文，详细分析了它与Batch Normalization的区别和在RNN中的应用。论文地址：https://arxiv.org/abs/1607.06450

- "The Illustrated Transformer"（Jay Alammar）— 一篇极具视觉效果的Transformer科普博客，用图解方式讲解Attention机制的内部运作。地址：https://jalammar.github.io/illustrated-transformer/

**相关工具与平台**：

- Lambda GPU Cloud（https://lambdalabs.com）— Karpathy推荐的GPU云服务商，适合训练中等规模模型
- Weights & Biases（https://wandb.ai）— 实验跟踪和可视化工具，推荐用于监控训练过程
- Hugging Face Transformers（https://github.com/huggingface/transformers）— 包含预训练GPT-2等模型的开源库

---

[^1^]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30.

[^2^]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*, 1(8), 9.

[^3^]: Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

[^4^]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

[^5^]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv preprint arXiv:1607.06450*.



---

## 8. 第8课：GPT Tokenizer — 文本与Token之间的桥梁

在前面的课程中，我们已经从零构建了一个完整的 GPT 模型 — 从 bigram 基线到多头自注意力机制，再到堆叠的 Transformer Block。但你可能注意到一个细节：我们一直使用字符级别的 tokenizer（character-level tokenizer），将每个字符直接映射为一个整数 ID。这种方式虽然简单直观，却与现代 LLM 的实际工作方式相去甚远。OpenAI 的 GPT 系列、Google 的 Gemini、Meta 的 Llama 等所有主流大语言模型，都使用一种更为复杂的分词系统，将文本切分为称为 **token** 的变长文本块。这些 token 可能是完整的英文单词（如 `"the"`、`"language"`），也可能是单词的一部分（如 `"token"`、`"ization"`），甚至是单个字符或标点符号。

这一课我们将深入探讨 tokenizer 的工作原理。Andrej Karpathy 在他的课程[^1^]中指出，tokenizer 是 LLM 流水线中一个独立且至关重要的阶段 — 它有自己的训练数据、自己的训练算法（Byte Pair Encoding，简称 BPE），以及一套完整的编码（encode）和解码（decode）机制。更重要的是，许多看似"模型能力不足"的 LLM 怪异行为 — 算术能力差、无法反向拼写单词、非英语语言表现不佳 — 实际上根源于 tokenizer 的设计，而非模型本身的缺陷。

我们将从零实现一个完整的 GPT 风格的 tokenizer，深入理解 BPE 算法的每一步，并探讨为什么 tokenizer 在带来效率提升的同时，也成为了 LLM 系统中一个难以根除的"痛点"。

### 8.1 Tokenizer的本质与重要性

#### 8.1.1 为什么需要Tokenizer：LLM处理的是token而非字符或单词

要理解 tokenizer 的必要性，我们首先需要思考一个基本问题：神经网络本质上是数学运算的集合 — 矩阵乘法、加法、非线性激活。它不能直接"阅读"文本字符串。因此，任何文本在输入模型之前，都必须被转换为数字序列。最简单的方案是字符级别映射：收集训练数据中所有唯一字符，为每个字符分配一个整数 ID。这种方式我们在前面的课程中已经使用过 — 对于 Shakespeare 数据集，字符表大约只有 65 个字符，vocab size 很小。

但字符级别的方案存在明显的效率问题。以英文为例，一个普通单词大约由 5-6 个字符组成。如果模型在字符级别上操作，预测一个单词就需要连续进行 5-6 次前向传播。更关键的是，模型需要从零学习字母组合成单词的规则 — 例如 `"t"` 后面经常跟着 `"h"`，这在大量文本中是显而易见的重复模式，却要在每次预测中重新计算。

那为什么不直接在单词级别操作呢？问题在于自然语言的词汇量极其庞大。英语中常用单词约有数万个，加上专有名词、技术术语、新造词和拼写变体，一个覆盖广泛的单词词汇表很容易膨胀到数十万甚至上百万。巨大的 vocab size 会直接导致模型最后的输出层（将隐藏状态映射到词汇表上每个词的概率）变得极其庞大和缓慢。而且，单词级别方案无法处理词汇表外的词（out-of-vocabulary，OOV） — 任何拼写错误、新造词或人名都会让模型束手无策。

Tokenizer 正是在这两个极端之间寻找平衡。它将文本切分为**子词单元**（subword units）— 既不是单个字符，也不是完整单词，而是介于两者之间的文本块。频繁出现的完整单词（如 `"the"`、`"and"`）成为独立的 token，而较少见的词则被拆分为更小的有意义的子词片段。这样既控制了 vocab size 在可管理的范围内（通常几万个 token），又能通过组合子词来表示任何新词，从根本上解决了 OOV 问题。

Karpathy 强调，token 是现代 LLM 的原子单位[^1^]。模型看到的不是 `"hello world"` 这 11 个字符，也不是两个单词，而可能是两个 token：`["hello", " world"]`（具体的切分取决于 tokenizer 的训练结果）。模型学习的是 token 与 token 之间的共现关系、语法结构和语义关联。理解这一点至关重要 — 当你看到 LLM 输出看似奇怪的结果时，很多时候问题的根源不在模型的"理解能力"，而在于输入文本被切分成了什么样的 token。

#### 8.1.2 Tokenizer是独立的训练阶段

Tokenizer 的另一个关键特性是：它是 LLM 流水线中**完全独立**的一个阶段。它有自己的训练数据集（通常是非常大量的文本语料），有自己的训练算法（BPE），训练完成后产生一套固定的编码和解码函数[^1^]。

具体来说，一个训练好的 tokenizer 包含三个核心组件：

1. **词汇表（Vocabulary）**：从 token ID 到对应字节序列的映射。例如，ID 256 可能对应字节序列 `b' t'`，ID 257 可能对应 `b'he'` 等。初始的 256 个 ID（0-255）对应所有可能的单字节值。

2. **合并规则（Merge Rules）**：记录了在 BPE 训练过程中哪些相邻 token 对被合并为新 token。这是一份有序列表 — 先合并的 pair 优先级更高。合并规则是 tokenizer 的"压缩算法"核心。

3. **编码函数 `encode()` 和解码函数 `decode()`**：`encode` 将字符串转换为 token ID 列表，`decode` 将 token ID 列表还原为字符串。这两个函数是 tokenizer 对外暴露的唯一接口。

这种独立性意味着：当你使用 GPT-4 时，它的 tokenizer 是预先训练好的、固定的。无论你是用它进行推理还是在其基础上微调，tokenizer 都不会改变。模型权重和 tokenizer 是两个分离的系统 — 模型权重通过梯度下降在大量文本上训练，而 tokenizer 通过 BPE 算法在（通常是另一批）文本上训练。

Karpathy 用一个清晰的图示表达了这个关系[^1^]：输入文本首先经过 tokenizer（`encode`）变成数字序列，然后输入 Transformer 模型进行处理；模型的输出（也是数字序列）再经过 tokenizer（`decode`）还原为可读的文本。Tokenizer 像一个"翻译官"，位于人类可读的文本和机器可处理的数字之间。

#### 8.1.3 LLM的怪异行为往往根源于Tokenization

在深入实现之前，让我们先看看 tokenizer 如何影响 LLM 的实际表现。Karpathy 在课程中列举了大量例子[^1^]，这些例子揭示了一个重要但常被忽视的真相：**许多 LLM 的"怪癖"不是模型智能不足，而是文本被切分成了对模型来说不够理想的 token**。

一个经典的例子是算术。问 GPT-4 `"3809 * 20092 = ?"`，它很可能会给出错误答案。但问题不在于模型不会乘法 — 如果数字被表示为单个数字字符，模型在大量训练后是有可能学会基本算术规则的。问题在于 `"3809"` 可能被切分为类似 `["380", "9"]` 或 `["38", "09"]` 这样的 token，而 `"20092"` 可能被切成完全不同的不规则组合。模型从未在"个位对齐相乘"的粒度上看过这些数字，因此它无法建立系统的算术推理能力。

另一个著名案例是反向拼写单词。如果你问 GPT-4 把 `"hello"` 倒过来写，它常常出错。原因很直接：在训练数据中，模型看到的是 token 序列如 `["hello"]` 或 `["hel", "lo"]`，而从未被要求将单词逐字符反转。Token 是原子 — 模型没有机会学习字符级别的操作。

对于非英语语言，问题更为严重。英语作为 tokenizer 训练数据中的主导语言，其高频单词通常能被高效地编码为单个 token。但一个中文句子可能被切成比英文多得多的 token，导致处理中文的"计算成本"远高于英文 — 同样长度的文本，中文可能需要 2-3 倍的 token 数量，这直接影响 API 调用费用和上下文窗口利用率。我们会在 8.4 节详细分析这些问题。

### 8.2 Byte Pair Encoding (BPE)算法

#### 8.2.1 BPE的核心思想：从字符开始，反复合并最常见的相邻token对

Byte Pair Encoding（字节对编码，简称 BPE）[^2^] 是一种从数据中学习词汇表的无监督分词算法。它的核心思想出奇地简单 — 从最小的单位（单个字节）出发，反复将训练数据中最频繁出现的相邻 pair 合并为新 token，直到词汇表达到预定大小。

让我们用一个具体例子来理解这个过程。假设我们要在以下简化文本上训练 tokenizer：

```
"aaabdaaabac"
```

初始状态下，每个字符就是一个独立的 token。我们将文本表示为字节（或字符）序列：

```
['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']
```

或者更精确地说，使用字符的 Unicode code point：

```
[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
```

（这里 97 是 `'a'` 的 ASCII/Unicode 编码，98 是 `'b'`，依此类推。）

第一步，我们统计所有相邻 pair 的出现频率。在这个序列中：
- `(97, 97)` 即 `('a', 'a')` 出现了 4 次（位置 0-1, 1-2, 5-6, 6-7）
- `(97, 98)` 即 `('a', 'b')` 出现了 2 次（位置 2-3, 7-8）
- `(98, 100)` 即 `('b', 'd')` 出现了 1 次
- 其他 pair 各出现 1 次

最频繁的 pair 是 `(97, 97)`。我们给它分配一个新 ID — 假设是 256（因为 0-255 已经被单个字节占用）。然后在整个序列中，将所有相邻的 `('a', 'a')` 替换为 256。序列变成：

```
[256, 97, 98, 100, 256, 97, 98, 97, 99]
```

注意这里的一个重要细节：替换是贪心且不回溯的。从左到右扫描，遇到 `('a', 'a')` 就替换为 256 并跳过这两个位置。原来的 `['a', 'a', 'a', 'b']` 变成了 `[256, 'a', 'b']` — 前两个 `'a'` 被合并，第三个 `'a'` 保留。

现在词汇表包含原来的 256 个字节 token 加上新 token 256，它代表字节序列 `b'aa'`。

第二步，在新的序列上再次统计 pair 频率。最频繁的 pair 可能变成了 `(256, 97)` 即 `('aa', 'a')`，出现 2 次。我们给它分配 ID 257，再次合并。

这个过程持续进行。每次合并都创造了一个新的、表示更长字节序列的 token。经过足够多的合并步骤后，词汇表中的 token 可以表示越来越长的文本片段 — 从常见字母组合如 `"th"`、`"ing"` 到完整的常用单词如 `"the"`、`"language"`。

BPE 这个名字来源于它的原始应用场景 — 数据压缩。将频繁出现的字节对替换为单字节标记，本质上就是在压缩数据。对于语言模型而言，这种"压缩"意味着模型可以用更短的序列长度来处理同样的文本，从而提高训练和推理效率。

#### 8.2.2 训练过程：统计频率 → 合并最频繁的pair → 更新语料 → 重复

现在让我们看看 BPE 训练的完整流程如何用代码实现。Karpathy 的 minBPE 项目[^3^]提供了一个极其清晰的实现，我们从中提取核心逻辑。

训练过程的第一步是将原始文本转换为初始 token 序列 — 也就是 UTF-8 字节序列。Python 的字符串编码功能让这一步非常直接：

```python
def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # 将文本编码为UTF-8字节序列
    text_bytes = text.encode("utf-8")  # raw bytes
    ids = list(text_bytes)             # 0到255之间的整数列表
```

这里 `text.encode("utf-8")` 将字符串转换为 Python 的 `bytes` 对象，然后 `list()` 将其展开为一个整数列表，每个整数在 0 到 255 之间。这就是我们的起始状态。

接下来是训练的核心循环。每次迭代做三件事：统计 pair 频率、找到最频繁的 pair、执行合并：

```python
    merges = {}  # (int, int) -> int: 记录pair到新token ID的映射
    vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes

    for i in range(num_merges):
        # 1. 统计所有相邻pair的出现频率
        stats = get_stats(ids)
        # 2. 找到出现次数最多的pair
        pair = max(stats, key=stats.get)
        # 3. 分配新的token ID
        idx = 256 + i
        # 4. 在整个序列中替换该pair
        ids = merge(ids, pair, idx)
        # 5. 保存合并规则并扩展词汇表
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
```

这段代码中的两个核心辅助函数 `get_stats()` 和 `merge()` 虽然简单，但承载了整个算法的精髓。`get_stats()` 遍历一个整数列表，统计每个相邻 pair 出现的次数：

```python
def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回每个连续pair的出现次数统计。
    示例: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # 遍历相邻元素
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

`zip(ids, ids[1:])` 是一个精妙的 Python 惯用法 — 它同时迭代列表的当前元素和下一个元素，产生所有相邻 pair。例如 `[1, 2, 3]` 产生 `(1, 2)` 和 `(2, 3)`。

`merge()` 函数负责实际执行替换操作：

```python
def merge(ids, pair, idx):
    """
    在整数列表ids中，将所有连续的pair替换为新整数token idx。
    示例: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果不是最后一个位置且pair匹配，就替换
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2  # 跳过两个已合并的位置
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

`merge()` 的关键在于它的线性扫描逻辑：从左到右遍历列表，每当发现匹配的 pair 就替换成新的 token ID 并前进两步；否则复制当前元素并前进一步。这个简单的过程在反复执行后，能产生惊人的压缩效果。

`vocab` 的构建也很有巧思。初始时 `vocab[idx]` 对每个 0-255 的 ID 存储对应的单字节 `bytes([idx])`。每次合并后，`vocab[idx] = vocab[pair[0]] + vocab[pair[1]]` — 新 token 对应的字节序列就是两个被合并 token 的字节序列的拼接。这确保了我们始终知道每个 token ID 背后代表的具体字节内容。

训练完成后，我们得到了两个核心数据结构：`merges`（有序的合并规则字典）和 `vocab`（完整的词汇表）。它们分别用于编码和解码过程。

#### 8.2.3 Vocabulary构建：训练结束时得到的token集合

当训练循环结束时，我们的词汇表 `vocab` 包含 `vocab_size` 个条目：前 256 个是所有可能的单字节值（0x00 到 0xFF），后续的每个条目都代表一个通过合并产生的多字节序列。例如，如果训练过程中先合并了 `('t', 'h')` 得到 token 256，再合并了 `(256, 'e')` 得到 token 257，那么 `vocab[257]` 就对应字节序列 `b'the'`。

这个词汇表的构建完全是确定性的 — 给定相同的训练文本和相同的 `vocab_size`，你总会得到相同的词汇表和合并规则。这是 BPE 的一个重要特性：编码和解码只需要保存 `merges` 字典和初始字节表，不需要存储整个词汇表（因为它可以确定性重建）。

一个有趣的观察是：vocab size 的选择直接影响 tokenizer 的行为。如果 vocab size 很小（比如 512），tokenizer 只能学习少数最常见的字节组合，大部分文本仍然会被拆分成较短的片段。如果 vocab size 很大（比如 GPT-4 使用的约 100,000），tokenizer 就能学习大量完整的单词和常见词组，将文本压缩成更短的 token 序列。OpenAI 在不同代的 GPT 模型中使用了不同的 vocab size — GPT-2 约 50,000，GPT-4 扩展到约 100,256[^3^]。

### 8.3 实现GPT Tokenizer

#### 8.3.1 基础Tokenizer：get_stats()统计频率、merge()执行合并

我们已经看到了 `get_stats()` 和 `merge()` 这两个核心函数的实现。现在让我们把它们组合成一个可用的 `BasicTokenizer`。这个基础版本不包含正则表达式分词，也不处理特殊 token — 它纯粹展示了 BPE 的核心逻辑[^3^]。

```python
from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        # 将token ID列表还原为字符串
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # 将字符串编码为token ID列表
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到在merges中优先级最高（索引最低）的可合并pair
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # 没有更多可合并的pair了
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
```

`decode()` 的逻辑非常直观：对于每个 token ID，从 `vocab` 中取出对应的字节序列，将所有字节拼接后做一次 UTF-8 解码就得到了原始字符串。这里使用了 `errors="replace"` 参数来处理可能的解码错误 — 虽然理论上 BPE 是无损的，但实际中某些字节组合可能不构成合法的 UTF-8 序列。

`encode()` 的过程则更有技巧性。你可能注意到，编码时我们不再是"找最频繁的 pair"，而是"找在 `merges` 中优先级最高（即合并索引最低）的可合并 pair"。为什么这么说？

关键在于**合并的顺序至关重要**。BPE 编码是确定性的 — 当面对一个从未见过的文本时，我们必须按照训练时确定的合并顺序来执行合并。具体来说，如果在训练时 pair A 比 pair B 更早被合并（即 A 的 merge rank 更低），那么在编码时，如果文本中同时出现了 A 和 B，我们必须先合并 A。

`encode()` 中的 `min(stats, key=lambda p: self.merges.get(p, float("inf")))` 这一行正是实现了这个逻辑。对于当前序列中的每一个 pair，查找它在 `merges` 中的排名（如果不在 `merges` 中则返回无穷大），选择排名最小的那个进行合并。这确保了编码过程与训练时的合并顺序一致。

这里有一个微妙但重要的终止条件：如果没有 pair 在 `self.merges` 中，那么 `min()` 会对所有 pair 返回 `float("inf")`，此时 `pair not in self.merges` 为真，循环终止。这个检测确保了我们不会在无法合并的情况下无限循环。

#### 8.3.2 Regex分词：GPT-2/GPT-4使用的预分词正则表达式

`BasicTokenizer` 虽然完整实现了 BPE 的核心逻辑，但它缺少一个关键步骤 — 预分词（pre-tokenization）。在实际应用中，GPT-2 和 GPT-4 在应用 BPE 之前，首先使用正则表达式将文本按类别分割成若干"块"（chunks），然后对每个块独立执行 BPE[^3^]。

为什么要这样做？想象一段混合了英文单词、数字和标点的文本：`"The price is $200.50 today."`。如果不做预分词，BPE 可能会学会将 `"$2"` 或 `"00."` 合并为一个 token — 但这种跨类别的合并在语言学上没有意义。一个数字和货币符号的组合不应该被当作一个独立的词汇单元来学习。预分词的目的就是确保 BPE 的合并不跨越类别边界。

GPT-2 使用的正则表达式分割模式如下[^3^]：

```python
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

这个正则表达式虽然看起来复杂，但分解开来逻辑很清晰：

- `'(?:[sdmt]|ll|ve|re)` — 匹配英文缩写形式，如 `'s`、`'d`、`'t`、`'ll`、`'ve`、`'re`（所有格和助动词缩写）
- ` ?\p{L}+` — 匹配可选前导空格后跟着一个或多个字母（Unicode letter 类别）
- ` ?\p{N}+` — 匹配可选前导空格后跟着一个或多个数字
- ` ?[^\s\p{L}\p{N}]+` — 匹配可选前导空格后跟着一个或多个非空白、非字母、非数字的字符（主要是标点符号）
- `\s+(?!\S)` — 匹配一串尾随空格（行尾的空格）
- `\s+` — 匹配其他空白字符序列

GPT-4 使用了更复杂的模式[^3^]：

```python
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

GPT-4 模式的改进包括：不区分大小写的缩写匹配（`'(?i:...)`），对数字的精细控制（`\p{N}{1,3}` 将数字按最多 3 位一组切分），以及对换行符的特殊处理（`\s*[\r\n]` 单独匹配换行序列）。这些改进使得 GPT-4 的 tokenizer 在处理代码和多语言文本时表现更好。

正则表达式分词对 BPE 训练流程的影响在于：文本首先被切分成若干块，然后每块被独立编码为字节序列，BPE 的统计和合并操作在每个块内部独立进行。这意味着 `"hello"` 和 `"world"` 之间的边界不会被跨越 — 即使 `"o w"` 在文本中频繁出现，它也不会被合并为一个 token。

#### 8.3.3 特殊Token处理：边界标记的添加

特殊 token 是 tokenizer 中另一个重要的概念。它们是预先定义好的、具有固定 ID 的 token，**不参与 BPE 合并过程**。最常见的特殊 token 是 `<|endoftext|>` — 用于标记文档边界，在训练时将多个文档拼接在一起，模型通过这个特殊标记知道前一个文档结束了[^3^]。

在 minBPE 的实现中，特殊 token 通过 `register_special_tokens()` 方法注册：

```python
def register_special_tokens(self, special_tokens):
    # special_tokens 是 str -> int 的字典
    # 示例: {"<|endoftext|>": 100257}
    self.special_tokens = special_tokens
    self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
```

特殊 token 的 ID 通常位于常规词汇表范围之外。例如，如果 vocab size 是 100,256，那么特殊 token 的 ID 从 100,256 开始。这些 ID 在编码时直接映射到对应的字符串，不参与任何 BPE 合并。

处理特殊 token 的编码逻辑需要特别注意 — 我们需要在文本中先识别出特殊 token 的位置，将文本切分成"普通文本段"和"特殊 token"交替的序列，然后只对普通文本段应用 BPE，直接保留特殊 token 的 ID[^3^]：

```python
def encode(self, text, allowed_special="none_raise"):
    # 根据allowed_special参数决定如何处理特殊token
    special = None
    if allowed_special == "all":
        special = self.special_tokens
    elif allowed_special == "none":
        special = {}
    elif allowed_special == "none_raise":
        special = {}
        # 如果文本中包含任何特殊token，报错
        assert all(token not in text for token in self.special_tokens)
    elif isinstance(allowed_special, set):
        special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

    if not special:
        return self.encode_ordinary(text)

    # 用正则表达式按特殊token切分文本
    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pattern, text)

    ids = []
    for part in special_chunks:
        if part in special:
            ids.append(special[part])  # 直接输出特殊token的ID
        else:
            ids.extend(self.encode_ordinary(part))  # 普通文本用BPE编码
    return ids
```

这里 `encode_ordinary()` 是不处理特殊 token 的基础编码方法 — 它只做 regex 分词和 BPE 合并。`encode()` 在此基础上增加了特殊 token 的处理层。

`allowed_special` 参数的设计考虑了不同场景的需求。`"none_raise"` 是安全模式 — 如果用户传入的文本中意外包含了特殊 token 字符串（例如用户输入中恰好有 `"<|endoftext|>"`），就直接报错提醒用户，避免特殊 token 被当作普通文本处理导致安全问题。`"all"` 则用于模型内部调用时，允许所有特殊 token 被正确识别。OpenAI 的 `tiktoken` 库也采用了类似的参数设计[^4^]。

#### 8.3.4 完整encode()和decode()的实现

现在我们来看完整的 `RegexTokenizer` 实现，它整合了 regex 预分词和特殊 token 处理[^3^]：

```python
import regex as re
from .base import Tokenizer, get_stats, merge

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 用正则表达式将文本切分为多个块
        text_chunks = re.findall(self.compiled_pattern, text)
        # 每块独立编码为字节序列
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # 统计所有块中的pair频率
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # 在每个块中独立执行合并
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # 对单个文本块执行BPE编码
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """忽略特殊token的编码"""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
```

`decode()` 方法处理两种情况：普通 token ID 从 `self.vocab` 中查找对应的字节序列，特殊 token ID 从 `inverse_special_tokens` 中查找对应的字符串，然后统一编码为 UTF-8 字节。所有字节拼接后做一次 UTF-8 解码得到最终文本。

训练流程中的关键变化是 `ids` 变成了一个列表的列表 — `ids = [list(ch.encode("utf-8")) for ch in text_chunks]`。每个文本块有自己的 ID 列表，统计时遍历所有块累积频率，合并时在每个块中独立执行。这确保了 BPE 合并不跨越 regex 定义的类别边界。

### 8.4 Tokenizer的深层问题

#### 8.4.1 为什么LLM做算术困难：数字被拆成不规则的token

现在我们来深入分析 tokenizer 导致的 LLM 行为怪癖。Karpathy 在课程中反复强调[^1^]：当你看到 LLM 在某些任务上表现不佳时，不要急于归因于模型的"推理能力不足"，而应该先看看输入文本被切分成了什么样的 token。

算术是最典型的例子。考虑简单的乘法 `"3809 * 20092"`。在 GPT-4 的 tokenizer 中，这个数字序列可能被切分成类似这样的 token[^1^]：

```
"3809"  -> ["380", "9"]
"20092" -> ["200", "92"]
```

（实际的切分可能因具体 tokenizer 版本而异，但核心问题是一致的：数字不是按单个数字字符切分，也不是按统一的位数切分，而是被切成了不规则的多位数片段。）

这意味着模型从未在训练数据中见过"个位对齐的竖式乘法"结构。在训练文本中，数字只是作为文本的一部分出现 — 日期、价格、统计数据 — 模型学到的是 `"380"` 后面经常跟着 `"9"` 这种统计规律，而非位值制（place-value system）的数学原理。当你问 `"3809 * 20092 = ?"` 时，模型实际上在做的是 token 序列的补全预测，而不是真正执行乘法运算。

对比来看，如果数字被切分为单个数字字符，`"3809"` 变成 `["3", "8", "0", "9"]`，那么模型至少有机会学习逐位运算的模式。即便在这种理想情况下，LLM 做多位数乘法仍然困难 — 因为这需要很长的推理链和精确的中间结果传递 — 但 tokenizer 的不规则切分让这个问题变得更加严重。

#### 8.4.2 多语言不平等：英语token少，其他语言token多，成本不对等

Tokenizer 训练数据中的语言分布直接影响不同语言的处理效率和模型表现[^1^]。BPE 算法会优先学习训练数据中最频繁出现的文本片段。由于 GPT 系列 tokenizer 的训练语料以英文为主导，高频英文单词通常能被高效地编码为单个 token，而其他语言的文本往往需要被拆分成更多 token。

让我们用一个具体例子来感受这种不平等。同样是表达"你好，世界"的意思：

- 英文 `"Hello, world!"` 可能被编码为 3-4 个 token
- 中文 `"你好，世界！"` 可能需要 6-8 个 token
- 日文 `"こんにちは、世界！"` 可能需要 8-12 个 token
- 韩语 `"안녕하세요, 세계!"` 可能需要 10-15 个 token

这种差异的后果是多方面的。首先，**经济成本** — OpenAI 等 API 提供商按 token 数量计费。同样长度的文本，用中文处理的直接费用可能是英文的 2-3 倍。其次，**上下文窗口利用率** — 如果一个语言模型的上下文窗口是 4096 个 token，一段中文文档能放入的有效信息量只有同长度英文文档的一半左右。最后，**模型表现** — 更长的 token 序列意味着注意力机制需要在更大的范围内计算关系，这对低资源语言是系统性的不利。

Karpathy 在课程中指出[^1^]，tokenizer 训练数据的语言偏见是 LLM 多语言能力差异的根本原因之一。即便模型架构本身是完全语言无关的，tokenizer 层面的不平等已经在"起跑线上"造成了差距。

#### 8.4.3 无法反向拼写：因为训练时从未被要求从token还原字符

LLM 无法可靠地将单词逐字符反向拼写，这个看似简单的任务对模型来说却出奇地困难[^1^]。原因同样在于 tokenizer 的原子性。

当你在提示词中说 `"Spell 'hello' backwards"`（将"hello"倒过来拼），模型看到的是 token 序列。假设 `"hello"` 恰好是一个完整的 token（ID 为 12345），那么模型需要输出 `"olleh"`。但 `"olleh"` 在 tokenizer 中可能根本不存在 — 因为 `"olleh"` 不是一个合法的英文单词。即使模型在概念上"知道" `"hello"` 的字母组成，它也必须通过输出一系列 token 来表达 `"olleh"`，而这些 token 可能是完全不熟悉的子词片段。

更深入的问题是，训练数据中几乎不存在"反向拼写"的任务。BPE 学到的 token 是正序文本的统计压缩 — `"th"`、`"he"`、`"the"` 这样的正序组合频繁出现，但 `"ht"`、`"eh"`、`"eht"` 这样的逆序组合在文本中极为罕见。因此模型缺乏学习逐字符操作的数据基础。

这个例子揭示了一个更普遍的 insight：LLM 的能力和缺陷很大程度上由训练数据的分布决定。Tokenizer 定义了模型"看到"世界的粒度，任何需要跨越这个粒度的操作 — 字符级别操作、精确字符串处理、需要逐位推理的数学计算 — 都会变得异常困难。

#### 8.4.4 未来的方向：可能的话删除Tokenizer这一层

既然 tokenizer 带来了这么多问题，一个自然的想法是：能否完全去掉这一层，让模型直接在原始字节（raw bytes）上操作？

Karpathy 在课程中明确表达了这个愿景[^1^]："理想情况下，我们应该删除 tokenizer 这一层。"如果模型能直接在 UTF-8 字节流上学习和生成，所有 tokenization 相关的问题都将迎刃而解 — 算术问题消失（每个数字字符都是独立的输入单元），多语言不平等消失（所有语言的字符在字节层面获得平等表示），反向拼写变得轻而易举（模型可以直接输出单个字节）。

这个方向的技术挑战在于**效率**。以字节为单位意味着序列长度会大幅增加 — 一段普通英文文本在 UTF-8 编码下，字节长度大约是字符数的 1 倍（ASCII 字符占 1 字节），而 token 数量通常只有字符数的 0.7-0.8 倍。更长的序列意味着更大的注意力计算开销（$O(n^2)$ 的复杂度增长）和更小的有效上下文窗口。

不过，Karpathy 也指出[^1^]，随着模型架构的演进 — 尤其是线性注意力机制（如 Mamba、RWKV 等 $O(n)$ 复杂度的注意力替代方案）和更长的上下文窗口技术的发展 — 在字节级别上训练大模型的可行性正在逐步提高。一些研究已经在这个方向上做出了探索，例如 ByteLaM 等模型尝试直接在字节上训练语言模型。虽然短期内 tokenizer 仍将是 LLM 系统的标配组件，但从长远来看，"删除 tokenizer"可能是让 LLM 更加通用和公平的重要一步。

### 8.5 课后练习与资源

#### 8.5.1 推荐资源

**核心代码与教程**

- Karpathy 的 minBPE 项目（https://github.com/karpathy/minbpe）[^3^] 是理解 GPT tokenizer 的最佳起点。这个项目包含了从 `BasicTokenizer` 到完整复刻 GPT-4 tokenizer 的全部代码，代码量精简但功能完整。项目附带的 `exercise.md` 提供了一套由浅入深的练习题，建议你按照练习题的指引，亲手实现一个能与 `tiktoken` 库输出一致的 GPT-4 tokenizer。

- Karpathy 的课程视频（https://www.youtube.com/watch?v=zduSFxRajkE）[^1^] 时长约 2 小时 13 分钟，是这一课最权威的学习材料。视频的前半部分专注于 BPE 算法的实现，后半部分深入讨论了 tokenizer 导致 LLM 行为怪癖的原因，以及多模态 tokenizer（图像、音频的向量化离散表示）的概念。

**可视化与实验工具**

- tiktokenizer Web 工具（https://tiktokenizer.vercel.app）是一个极佳的可视化工具，让你可以实时观察不同文本如何被 GPT-2、GPT-4 和 Claude 的 tokenizer 切分。试着输入包含数字、中文、代码和特殊符号的文本，观察不同 tokenizer 的切分策略差异。这个工具对理解"LLM 看到什么"非常有帮助。

- OpenAI 的 tiktoken 库（https://github.com/openai/tiktoken）[^4^] 是 OpenAI 官方发布的 Python tokenizer 库，提供了高性能的 tokenizer 实现（底层使用 Rust 加速）。你可以用它来做对比实验 — 在实现自己的 tokenizer 后，用 `tiktoken` 验证输出是否一致。

**其他Tokenizer实现**

- Google 的 SentencePiece 库（https://github.com/google/sentencepiece）[^5^] 是另一个广泛使用的 tokenizer 训练工具，被 Llama 2 等模型采用。与 GPT 的 BPE tokenizer 不同，SentencePiece 直接从原始文本（不需要预分词）训练，支持 BPE 和 Unigram 两种算法。如果你有兴趣训练自己的多语言 tokenizer，SentencePiece 是一个更灵活的选择。

- OpenAI 原始的 GPT-2 tokenizer 代码（https://github.com/openai/gpt-2/blob/master/src/encoder.py）[^6^] 展示了 GPT-2 时代 tokenizer 的实现方式。对比 minBPE 的代码和这段原始代码，你可以看到 Karpathy 如何用更清晰的结构实现了相同的功能。

**学术论文**

- Sennrich 等人的论文 "Neural Machine Translation of Rare Words with Subword Units"（https://arxiv.org/abs/1508.07909）[^2^] 首次将 BPE 算法引入 NLP 领域，奠定了子词分词的理论基础。

- GPT-2 论文 "Language Models are Unsupervised Multitask Learners"（https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf）[^6^] 描述了 GPT-2 tokenizer 的设计，包括字节级别 BPE（byte-level BPE）的方案。

**课后练习**

1. **实现完整的 GPT-4 Tokenizer**：参考 minBPE 项目的 `exercise.md`，从零实现一个 `GPT4Tokenizer` 类，使其输出与 `tiktoken` 库中的 GPT-4 tokenizer 完全一致。这是本课最具挑战也最有价值的练习。

2. **观察不同语言的 token 效率**：使用 tiktoken Web UI 或 Python 库，收集多语言文本样本（英文、中文、日文、阿拉伯文等），统计每种语言平均每个字符需要多少个 token。绘制对比图表，量化分析 tokenizer 的多语言偏见。

3. **探索 tokenizer 导致的 LLM 行为**：设计一组测试用例（多位数乘法、反向拼写、空格敏感性测试等），在 GPT-4 或其他 LLM API 上运行，记录哪些任务模型表现不佳。然后用 tiktokenizer 观察失败案例中输入文本的 token 切分方式，分析 tokenizer 与模型表现之间的关联。

4. **训练你自己的 tokenizer**：选择一个特定领域的大量文本（如代码仓库、法律文档、医学文献），用 minBPE 训练一个 vocab size 为 8192 或 16384 的 tokenizer。观察训练过程中最先被合并的是哪些 pair — 它们反映了该领域的哪些高频模式？

5. **思考"无 tokenizer"方案**：调研近年来在字节级别上训练语言模型的研究工作（如 ByteLaM、ByT5 等），撰写一篇简短的调研笔记，分析这些方案的优势、局限性和技术挑战。



---

# 第9课：State of GPT — 大语言模型的全景图

前面的课程中，我们从最基础的多层感知机开始，逐步搭建了自注意力机制、完整的Transformer块，最终训练出了能生成连贯文本的GPT模型。但我们所复现的，本质上只是GPT训练流程中第一个阶段（Pretraining）的产出物——一个**Base Model（基础模型）**。它像是一只"训练有素的鹦鹉"：能流利地续写文本，却不受控制，也不懂得扮演一个有用的助手。

这节课的内容来自Andrej Karpathy在Microsoft Build 2023上的演讲[^1^]。他将带我们站在更高的视角，俯瞰GPT模型从互联网 raw text 到最终成为ChatGPT这样有用助手的完整旅程。这个旅程包含四个阶段：Pretraining（预训练）、Supervised Fine-Tuning（监督微调）、Reward Modeling（奖励模型训练）和RLHF（人类反馈强化学习）。理解这四阶段流程，是理解现代大语言模型（Large Language Model，LLM）的钥匙。

---

## 9.1 GPT训练的四阶段流程

Karpathy将整个训练流程比喻为"从互联网文本中训练出一只鹦鹉"。这个比喻的精妙之处在于：Pretraining阶段确实让模型学会了"说话"——它记住了海量文本中的统计规律，能生成流畅的文本；但它并不理解什么是有用的回答，也不知道何时该停止。后续的三个阶段，本质上都是在解决一个问题：**如何把这个"鹦鹉"变成一个"有用的助手"**。

让我们用一个简单的代码片段来建立直觉——下面是一个模拟四阶段训练流程的骨架，它展示了每个阶段的输入、输出和核心目标。我们将在后续小节中逐一深入每个阶段。

```python
# GPT助手训练的四阶段流程（概念框架）

# 阶段1: Pretraining —— 在互联网文本上学习下一个token预测
base_model = pretrain(
    data="海量互联网文本 (Common Crawl, WebText, Books)",
    objective="下一个token预测: P(next_token | previous_tokens)",
    compute="最大 (~99%的总计算量)",
    output="Base Model —— 强大的通用表示，但不受控"
)

# 阶段2: Supervised Fine-Tuning —— 用高质量对话数据训练助手行为
sft_model = supervised_finetune(
    base_model=base_model,
    data="人工标注的高质量对话 (question-answer pairs)",
    objective="继续下一个token预测，但数据是对话格式",
    output="SFT Model —— 学会了助手角色的对话格式"
)

# 阶段3: Reward Modeling —— 训练一个能评估回答质量的模型
reward_model = train_reward_model(
    data="人类对多个回答的排序数据 (A > B > C)",
    objective="预测人类偏好: score(answer) 应该与人类排序一致",
    output="Reward Model —— 给回答打分的评判器"
)

# 阶段4: RLHF —— 用强化学习优化模型输出
final_model = rlhf(
    sft_model=sft_model,
    reward_model=reward_model,
    algorithm="PPO (Proximal Policy Optimization)",
    objective="生成能获得高reward分数的回答",
    output="RLHF Model —— 对齐人类偏好的最终助手"
)
```

这个骨架看似简洁，但每个阶段背后都有深刻的工程学考量和理论支撑。让我们逐一拆开来看。

### 9.1.1 Pretraining：在海量文本上学习语言模型的基础能力

Pretraining是整个流程中计算量最大的阶段——它消耗了约99%的训练计算资源。但有趣之处在于，它的目标却出奇地简单：**预测下一个token**。

给定一段文本序列 $x_1, x_2, \ldots, x_T$，模型被训练来最大化以下似然函数：

$$\mathcal{L}_{\text{pretrain}} = \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

这个目标函数的本质是什么？它要求模型在看过前面的所有token后，准确地预测下一个token是什么。为了达到这个目标，模型被迫学习语言的结构、世界的知识、推理的模式——因为所有这些信息都有助于更好地预测下一个词。

Karpathy用一个生动的比喻来描述这个过程：预训练就像让模型**读完了整个互联网**。它在这个过程中形成了对世界的深刻理解——语法规则、事实知识、推理链条、甚至是不同文体的风格。这个理解被压缩进了模型数十亿参数的权重矩阵中。

Pretraining的数据来源极其广泛：Common Crawl（数十亿网页的爬取数据）、WebText（高质量网页文本）、书籍（Books1和Books2）、Wikipedia（结构化知识）等。以GPT-3为例，其训练数据总量约3000亿token，涵盖了人类知识的方方面面。

Pretraining的输出被称为**Base Model（基础模型）**。它具备了强大的通用表示能力（powerful, general representations）——即使只是做下一个token预测，它也能表现出翻译、问答、代码生成、文本总结等能力。但你很快就会发现问题：当你向一个base model提问时，它不会回答你的问题，而是会继续生成类似的问题，或者开始一段开放式的叙述。就像一只聪明的鹦鹉，它会继续说下去，但不受你的控制。

### 9.1.2 Supervised Fine-Tuning (SFT)：用高质量对话数据训练模型遵循指令

从Base Model到有用的助手，关键的第一步是**Supervised Fine-Tuning（监督微调，简称SFT）**。这个阶段的目标非常直观：让模型学习**对话格式**和**助手角色**。

具体怎么做？研究者会准备大量高质量的人工标注对话数据——通常是问题-答案对（question-answer pairs）。例如：

```
User: 解释牛顿第二定律。
Assistant: 牛顿第二定律指出，物体的加速度与作用在其上的合外力成正比，与物体的质量成反比...
```

SFT的方法在技术上很简单：将预训练好的base model在对话数据上继续训练，目标函数仍然是下一个token预测。但数据的格式变了——不再是互联网的raw text，而是结构化的对话记录。为了让模型区分不同角色，通常会在问题和答案前后加上特殊的token标记，比如 `<|user|>` 和 `<|assistant|>`。

Karpathy将SFT比喻为**"角色扮演"**。模型在预训练阶段读了整个互联网，其中包括无数的问答、教程、对话记录。SFT本质上是在告诉模型："现在我要你扮演一个有用的助手。这是助手的说话方式，这是回答问题的格式。"模型学会了在输入中识别用户的问题，然后生成对应的回答。

SFT的输出我们称之为**SFT Model**。它已经具备了基本的助手能力：能理解指令、遵循格式、给出有用的回答。但SFT alone is not enough——它有两个关键局限：

第一，人类的偏好难以用精确的数学语言表达。什么是"更好的回答"？更简洁？更详细？更幽默？更严谨？这些偏好是微妙的、情境依赖的，无法通过简单的监督学习完全捕捉。

第二，SFT模型在面对open-ended generation时，可能会产生冗长、重复、或者风格不一致的回答。它学会了"回答的格式"，但还没有学会"什么是高质量的回答"。

### 9.1.3 Reward Modeling (RM)：训练模型评估回答质量的能力

为了解决SFT的局限，我们需要一种方式来量化"回答质量"。直接的思路是：让人类来评判。但人类评判者不可能实时参与每一个回答的评估——那样太慢、太贵。所以，我们训练一个模型来**模拟人类的评判**——这就是**Reward Model（奖励模型，简称RM）**。

Reward Modeling的数据收集过程是这样的：对于同一个问题，让SFT模型生成多个不同的回答（通常是4到9个变体），然后请人类标注者对这些回答进行**排序**。例如：

```
问题: 如何学习Python编程?
回答A: 从基础语法开始，推荐《Python编程：从入门到实践》... (排序: 1)
回答B: 去网上找教程。 (排序: 3)  
回答C: Python是一种高级编程语言，由Guido van Rossum于1991年创建... (排序: 2)
```

注意这里的关键设计：人类只需要做**相对比较**（A比B好），而不需要做**绝对评分**（给A打8分）。研究表明，人类做相对比较的一致性远高于绝对评分。

有了这些排序数据，我们如何训练Reward Model？核心思想是：训练一个模型，让它输出的**标量分数**与人类的排序一致。具体来说，对于同一问题的两个回答 $y_i$ 和 $y_j$，如果人类认为 $y_i$ 优于 $y_j$，那么reward model应该满足 $r_\theta(x, y_i) > r_\theta(x, y_j)$。

损失函数通常采用**Bradley-Terry模型**的形式，本质上是将排序问题转化为一个分类问题：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

这里 $y_w$ 是标注者偏好的回答（win），$y_l$ 是较差的回答（loss），$\sigma$ 是sigmoid函数。这个损失函数的含义是：如果reward model给win的回答打的分数高于loss的回答，损失就小；反之则大。

Reward Model通常基于同一个base model架构，只是最后的输出层从预测vocab分布变成了一个标量分数。训练完成后，它成了一个"自动评判器"：给定一个问题和一个回答，它能输出一个数值表示这个回答的"质量"。

### 9.1.4 Reinforcement Learning from Human Feedback (RLHF)：用PPO优化模型输出

现在我们有了三个组件：SFT Model（能生成回答）、Reward Model（能评判回答质量），以及一个明确的目标——生成能获得高reward分数的回答。这正是一个**强化学习（Reinforcement Learning，RL）**问题的标准设定。

RLHF使用**PPO（Proximal Policy Optimization，近端策略优化）**算法来优化SFT模型。PPO的核心思想可以概括为：在当前策略（即当前的语言模型）附近，以小步长进行探索，找到能产生更高reward的更新方向，同时避免策略发生剧烈变化。

在RLHF的框架中：

- **策略（Policy）**：就是我们的语言模型 $\pi_\theta$，它接收一个prompt $x$，输出一个token序列 $y$。
- **环境（Environment）**：语言模型生成token的过程——每生成一个token，就将它附加到上下文中，继续生成下一个。
- **奖励函数（Reward Function）**：由Reward Model提供，对完整生成的回答打分 $r_\theta(x, y)$。
- **目标**：最大化期望奖励。

PPO的优化目标函数如下：

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x, y) \sim \pi_{\theta_{\text{old}}}} \left[ \min \left( \rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中 $\rho_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$ 是新策略与旧策略的概率比，$A_t$ 是advantage（优势函数），$\epsilon$ 是一个很小的超参数（通常是0.1或0.2）。clip操作确保了策略的更新不会太大——如果概率比超出了 $[1-\epsilon, 1+\epsilon]$ 的范围，梯度就会被截断。

但RLHF有一个著名的问题需要特别注意：**Mode Collapse（模式坍塌）**。

模式坍塌描述的是这样一种现象：模型可能找到某种"捷径"来欺骗reward model，而不是真正产生高质量的回答。例如，模型可能发现某种特定的开头格式（如"当然！我很乐意帮助您"）能获得较高的reward分数，于是它在所有回答中都使用这种格式，即使并不合适。更严重的情况是，模型可能产生看似合理但实则空洞、重复的内容，因为这些内容恰好迎合了reward model的偏好。

缓解模式坍塌的关键技术是**KL散度约束**。在RL训练过程中，除了最大化reward，还要约束当前策略 $\pi_\theta$ 不要偏离SFT模型 $\pi_{\text{SFT}}$ 太远：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} - \beta \cdot D_{\text{KL}}\left( \pi_\theta \,\|\, \pi_{\text{SFT}} \right)$$

其中 $\beta$ 是一个超参数，控制着"偏离容忍度"。KL散度项确保了模型在追求高reward的同时，不会忘记在SFT阶段学到的基本语言能力。这就像是给模型戴上了一个"缰绳"：你可以优化，但不能跑得太远。

让我们用一个简化的代码示例来理解RLHF的训练循环：

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class RLHFTrainer:
    """简化的RLHF训练器，展示核心思想"""
    
    def __init__(self, sft_model, reward_model, beta=0.1, epsilon=0.2):
        # 策略模型（可训练）从SFT模型初始化
        self.policy = sft_model  # 这是我们要优化的目标
        self.policy_old = copy.deepcopy(sft_model)  # 旧的策略快照
        
        # Reward Model（冻结参数，不参与训练）
        self.reward_model = reward_model
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        self.beta = beta      # KL散度权重
        self.epsilon = epsilon  # PPO clip参数
        self.optimizer = Adam(self.policy.parameters(), lr=1e-5)
    
    def compute_reward(self, prompts, responses):
        """使用Reward Model计算回答的reward分数"""
        with torch.no_grad():
            rewards = self.reward_model(prompts, responses)
        return rewards.squeeze(-1)  # (batch_size,)
    
    def ppo_step(self, prompts, old_responses, advantages):
        """一次PPO更新步骤"""
        # 1. 新策略生成token的概率
        new_logprobs = self.policy.get_token_logprobs(prompts, old_responses)
        
        # 2. 旧策略生成token的概率
        old_logprobs = self.policy_old.get_token_logprobs(prompts, old_responses)
        
        # 3. 概率比
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 4. PPO clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # 5. KL散度惩罚（防止偏离SFT模型太远）
        kl_div = (new_logprobs - self.sft_logprobs).mean()
        
        # 6. 总损失 = PPO损失 + KL惩罚
        loss = ppo_loss + self.beta * kl_div
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

这个简化的实现捕捉了RLHF的核心思想：策略模型生成回答，reward model打分，PPO算法根据分数更新策略，同时KL散度约束防止模型坍塌。在实际系统中（如OpenAI的InstructGPT），这个流程会在数千个prompt上迭代多轮，逐步将模型对齐到人类偏好。

---

## 9.2 从GPT-1到GPT-4的能力演进

理解了训练流程后，我们自然会问一个问题：当模型变得更大、训练得更久时，会发生什么？这个问题的答案揭示了LLM最迷人的一面。

### 9.2.1 规模定律（Scaling Laws）：模型参数量、数据量、计算量与性能的关系

2020年，OpenAI的研究团队发表了一篇里程碑式的论文，揭示了一个惊人的规律：**语言模型的性能与三个因素之间存在幂律关系**——模型参数量（$N$）、训练数据量（$D$）和计算量（$C$）。

具体来说，在足够大的范围内，模型的测试损失（test loss）满足：

$$L(N, D) \propto \frac{1}{N^\alpha} + \frac{1}{D^\beta}$$

其中 $\alpha$ 和 $\beta$ 是幂指数（通常在0.05到0.1之间）。这个公式告诉我们两个关键信息：

第一，**模型越大，性能越好**——在固定数据量的情况下，增加参数量能持续降低损失。

第二，**数据越多，性能越好**——在固定模型大小的情况下，增加训练数据也能持续降低损失。

这个规律被证明在多个数量级上都成立——从几百万参数到数千亿参数。它像是一个"物理定律"，给了我们预测大模型行为的数学工具。例如，如果我们知道一个100M参数的模型在某个任务上达到了某个性能，我们就可以大致预测1B参数和10B参数模型在同一任务上的表现。

Scaling Laws的实践意义是巨大的。它指导了GPT系列的发展策略：GPT-1（1.17亿参数）、GPT-2（15亿参数）、GPT-3（1750亿参数）、GPT-4（参数量未公开，估计在万亿级别）。每一代都在遵循这个"更大即更好"的规律，同时也在探索更高效的训练方式和更好的数据质量。

但Scaling Laws揭示的不仅仅是"更大更好"这个单调趋势。随着规模的增长，一些意想不到的事情开始发生——这引出了**Emergence（涌现）**这个概念。

### 9.2.2 Emergence：某些能力只在模型达到一定规模后才突然出现

Emergence是LLM领域最令人着迷的现象之一。它描述的是：**某些能力并不会随着模型规模的增大而平稳提升，而是在达到某个临界规模后突然"涌现"**。

以下是一些经典的涌现能力例子：

**算术能力**。小模型（< 10B参数）在做多位数加法时表现几乎和随机猜测一样差。但当模型规模超过某个阈值后，它突然就能正确完成多位数的加、减、乘、除运算。这不是因为训练数据中包含了更多的算术题（事实上，互联网上的算术题并不多），而是因为大模型自发地学会了处理数字的算法。

**链式推理（Chain-of-Thought Reasoning）**。当模型足够大时，一个神奇的技巧开始奏效：在prompt中加入"Let's think step by step"（让我们一步一步思考），模型的推理准确率会大幅提升。小模型对此无感，大模型则能自发地分解问题、逐步求解。

**上下文学习（In-Context Learning）**。这是大模型最令人惊叹的能力之一：你只需要在prompt中给出几个示例（few-shot），模型就能理解任务模式并正确执行——无需任何参数更新。例如：

```
英语 -> 中文
cat -> 猫
dog -> 狗
bird -> 鸟
fish -> 
```

大模型能正确回答"鱼"，即使它从未专门被训练过翻译这对语言。这种"从示例中学习"的能力在大模型中才显著出现。

**指令遵循（Instruction Following）**。大模型能理解和执行用自然语言描述的指令，即使这些指令的组合方式在训练数据中从未出现过。例如："用莎士比亚的风格写一段关于人工智能的十四行诗"——这个具体请求在训练数据中几乎不可能出现，但模型能将其分解为"莎士比亚风格"+"十四行诗格式"+"人工智能主题"三个子任务并组合执行。

Emergence现象的科学解释至今仍是开放性问题。一种直觉性的理解是：小模型可能学会了语言的"表面统计"——词与词之间的共现关系；而大模型在压缩海量文本的过程中，被迫学到了更深层的**抽象表示**和**推理规则**。这些高层能力的"涌现"，某种程度上类似于复杂系统中相变的发生——当系统的复杂度超过某个临界点后，全新的宏观行为模式自发形成。

从GPT-1到GPT-4的演进史，就是一部能力涌现的编年史。GPT-1展示了Transformer在语言建模上的潜力；GPT-2展示了zero-shot任务执行能力；GPT-3展示了few-shot上下文学习的惊人效果；GPT-4则在推理、代码、多模态理解等多个维度上实现了质的飞跃。

### 9.2.3 Tool Use：LLM调用外部工具（API、搜索、代码执行）的能力

模型规模增大带来的另一个重要能力进化是**Tool Use（工具使用）**。这个概念的直觉很清晰：LLM不应该仅仅依赖它"记住"的参数化知识（parametric knowledge），还应该能够使用外部工具来获取信息、执行计算、与外部世界交互。

为什么Tool Use如此重要？因为LLM本身有一些根本性的局限：

第一，**知识的时效性**。模型的知识被"冻结"在预训练数据的时间点。GPT-4知道2024年4月之前发生的事，但对之后的事件一无所知。如果能让模型调用搜索引擎，它就能获取最新的信息。

第二，**计算的精确性**。LLM本质上是一个统计匹配引擎，不擅长精确计算。让它计算"367 × 482"可能会出错，但如果能让它调用计算器，结果就能保证精确。

第三，**代码执行**。模型可以生成代码，但无法直接执行代码来验证正确性。如果能让模型在代码解释器中运行代码，它就能进行精确的数值计算、数据处理、甚至自调试。

Tool Use的实现方式 elegantly simple：LLM被训练来发出**特殊token**，外围系统检测到这些token后调用相应的工具，再将工具的输出结果返回给LLM。例如：

```python
# Tool Use的核心机制：LLM通过特殊token调用外部工具

class ToolUseSystem:
    """LLM工具使用系统的简化实现"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            "<|BROWSER|>": self.web_search,      # 网络搜索
            "<|CALCULATOR|>": self.calculate,    # 精确计算
            "<|CODE|>": self.execute_code,       # 代码执行
            "<|DALLE|>": self.generate_image,    # 图像生成
        }
    
    def run(self, user_query):
        """主循环：LLM生成 -> 检测工具调用 -> 执行 -> 返回结果"""
        context = f"User: {user_query}\nAssistant:"
        
        while True:
            # LLM生成后续文本
            response = self.llm.generate(context, max_tokens=100)
            
            # 检测是否包含工具调用token
            tool_called = False
            for tool_token, tool_func in self.tools.items():
                if tool_token in response:
                    # 提取工具参数
                    query = response.split(tool_token)[1].strip()
                    
                    # 执行工具
                    tool_result = tool_func(query)
                    
                    # 将结果追加到上下文，让LLM继续生成
                    context += response + f"\n[Tool Result]: {tool_result}\n"
                    tool_called = True
                    break
            
            if not tool_called:
                # 没有工具调用，直接返回回答
                return response
    
    def web_search(self, query):
        # 实际实现中调用搜索引擎API
        return f"搜索结果: ..."
    
    def calculate(self, expression):
        try:
            return str(eval(expression))  # 安全评估
        except:
            return "计算错误"
    
    def execute_code(self, code):
        # 实际实现中在安全沙箱中运行代码
        return "代码执行结果: ..."
    
    def generate_image(self, prompt):
        # 实际实现中调用DALL-E等图像生成API
        return "[图像数据]"

# 使用示例
system = ToolUseSystem(llm=my_llm)
result = system.run("2024年诺贝尔物理学奖得主是谁？同时帮我计算一下 367 * 482")
# LLM会: 1) 发出<|BROWSER|>搜索最新信息
#        2) 发出<|CALCULATOR|>计算精确数值
#        3) 综合结果给出最终回答
```

Tool Use将LLM从一个"封闭的知识库"转变为一个"开放的推理引擎"。它是GPT-4能执行复杂多步任务的关键技术之一，也是当前LLM应用架构（如Agent系统）的基石。

---

## 9.3 LLM应用架构模式

理解了LLM的训练流程和能力演进后，我们来看看在实际应用中如何构建基于LLM的系统。当前业界已经形成了几种成熟的架构模式。

### 9.3.1 检索增强生成（RAG）：结合外部知识库减少幻觉

**幻觉（Hallucination）**是LLM最著名的问题之一：模型会自信地生成看似合理但实际上错误的内容。它并非"故意"欺骗，而是在做它唯一被训练做的事——生成统计上合理的文本序列。当模型不确定时，它不会说"我不知道"，而是会"编造"一个最可能的答案。

Karpathy在他的后续视频中有一个精妙的重要观点：**Finetuning不会"修复"幻觉**。SFT和RLHF本质上只是将base model的"梦境"引导为"有用的助手梦境"——模型依然在做梦，只是梦的内容变得更配合、更有用了。减少幻觉的根本方法不是通过训练，而是给模型提供**可靠的信息来源**。

**RAG（Retrieval-Augmented Generation，检索增强生成）**正是基于这个思路。它的核心思想是：在让模型生成回答之前，先从外部知识库中检索相关的文档，将这些文档作为上下文（context）提供给模型，让模型基于这些可靠信息来生成回答。

RAG的工作流程如下：

```python
# RAG系统的核心流程

class RAGSystem:
    """检索增强生成系统"""
    
    def __init__(self, llm, vector_db):
        self.llm = llm
        self.vector_db = vector_db  # 向量数据库
    
    def answer(self, user_query, top_k=3):
        # 步骤1: 将查询编码为向量
        query_embedding = embed(user_query)
        
        # 步骤2: 在向量数据库中检索最相似的文档
        retrieved_docs = self.vector_db.similarity_search(
            query_embedding, 
            top_k=top_k
        )
        
        # 步骤3: 将检索到的文档组装为上下文
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        
        # 步骤4: 构建prompt，要求模型基于上下文回答
        prompt = f"""基于以下参考资料回答问题：

参考资料：
{context}

问题：{user_query}

请仅基于以上参考资料回答。如果资料中没有相关信息，请明确说明。"""
        
        # 步骤5: LLM基于检索到的信息生成回答
        response = self.llm.generate(prompt)
        return response
```

RAG的价值在于它将模型的"参数化知识"（存储在权重中、可能有偏差的记忆）与"外部知识"（可靠、可验证、可更新的文档）分离开来。当检索到的文档进入模型的上下文窗口（context window）后，它们成为了模型的"工作记忆"（working memory）——模型可以直接引用这些文本中的事实，而不是依赖参数中可能不准确的知识。

Karpathy特别提醒：**如果LLM使用了浏览或检索工具，答案进入了上下文窗口的"工作记忆"，可信度会远高于仅凭记忆生成的内容**。这是一个区分LLM回答可靠性的重要mental model。

RAG系统的关键组件包括：

- **Embedding模型**：将文本转换为向量表示，使得语义相似的文本在向量空间中距离更近。
- **向量数据库**：存储文档的embedding向量，支持高效的相似性搜索（如FAISS、Pinecone、Weaviate等）。
- **检索策略**：可以是简单的向量相似性搜索，也可以是结合关键词匹配、重排序（reranking）等更复杂的多阶段检索。
- **重排序器（Reranker）**：在初步检索后，用一个更精确的模型对候选文档重新排序，提高检索质量。

### 9.3.2 Agent架构：LLM作为推理引擎，自主规划和执行多步任务

如果说RAG是"给模型提供知识"，那么**Agent（智能体）架构**就是"给模型赋予行动力"。Agent的核心思想是：将LLM作为一个**推理引擎（reasoning engine）**，它能够自主分解复杂任务、规划执行步骤、调用工具、观察结果、并据此调整后续行动。

一个典型的Agent架构遵循**ReAct（Reasoning + Acting）**框架，它的运作方式是循环式的：

```python
# ReAct Agent的简化实现

class ReActAgent:
    """
    ReAct Agent: 推理(Reasoning) + 行动(Acting) 的循环
    遵循 thought -> action -> observation -> thought -> ... 的循环
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools  # 可用的工具字典
        self.max_iterations = 10  # 防止无限循环
    
    def run(self, task):
        # 初始化思考上下文
        context = f"你需要完成以下任务：{task}\n\n"
        context += "你可以使用以下工具：\n"
        for name, tool in self.tools.items():
            context += f"- {name}: {tool.description}\n"
        context += "\n请按照以下格式思考：\n"
        context += "Thought: 你的思考过程\nAction: 工具名[参数]\n"
        context += "Observation: 工具返回的结果\n\n开始！\n"
        
        for step in range(self.max_iterations):
            # LLM生成下一步的思考和行动
            response = self.llm.generate(context, stop=["Observation:"])
            context += response
            
            # 解析行动
            if "Action:" in response:
                action_str = response.split("Action:")[1].strip().split("\n")[0]
                tool_name, tool_input = self._parse_action(action_str)
                
                # 执行工具
                if tool_name in self.tools:
                    observation = self.tools[tool_name].run(tool_input)
                    context += f"\nObservation: {observation}\n"
                else:
                    context += f"\nObservation: 错误：未知工具 {tool_name}\n"
            
            # 检查任务是否完成
            if "Final Answer:" in response:
                return response.split("Final Answer:")[1].strip()
        
        return "达到最大迭代次数，任务未完成。"
    
    def _parse_action(self, action_str):
        """解析 'tool_name[input]' 格式的行动"""
        tool_name = action_str.split("[")[0].strip()
        tool_input = action_str.split("[")[1].rstrip("]").strip()
        return tool_name, tool_input
```

ReAct的核心洞察在于：**将推理过程显式地表示为文本**，让LLM在生成回答的同时，也生成自己的"思考链"。这些思考文本被保留在上下文中，形成了模型的"工作记忆"，使得多步推理成为可能。

Agent架构的发展已经从简单的ReAct演进到了更复杂的系统，例如：

- **Plan-and-Execute**：先制定完整计划，再逐步执行。
- **Multi-Agent系统**：多个专门的Agent协作完成复杂任务（如一个Agent负责研究，一个负责编码，一个负责审查）。
- **Reflection Pattern**：Agent在执行后回顾自己的表现，从错误中学习并改进。

Agent架构的一个关键洞见来自Karpathy的观察：**模型需要token来思考**。LLM的推理能力在很大程度上取决于它有多少"思考空间"——即生成token的预算。给一个模型足够多的token来展开推理过程，它的表现会显著优于被要求立即给出答案的情况。这解释了为什么Chain-of-Thought prompting如此有效：它本质上是在为模型争取更多的思考token。

### 9.3.3 多模态扩展：从文本到图像、音频、视频的统一理解

GPT系列最初是纯文本模型，但LLM的能力边界正在快速扩展到其他模态。**多模态（Multimodality）**指的是模型能够同时理解和生成多种类型的数据——文本、图像、音频、视频。

多模态LLM的实现思路是将不同模态的数据统一转换为"token序列"，然后用同一个Transformer架构来处理。以视觉为例：

- **图像编码**：使用Vision Encoder（如ViT、CLIP的视觉塔）将图像转换为一系列视觉token。
- **统一序列**：视觉token与文本token混合在同一个序列中，用特殊的token类型标记来区分。
- **统一训练**：在混合了文本、图像-文本对、视频-文本对等数据的多模态语料上继续训练。

GPT-4V（Vision）展示了令人印象深刻的多模态能力：它能理解图片内容、分析图表、解读手写笔记、甚至理解梗图（meme）中的幽默。这种能力的本质仍然是next token prediction——只不过"token"不再局限于文本，还包括了视觉token。

Karpathy在他的视频中将LLM比喻为一个**"新兴的操作系统（LLM OS）"**。就像传统操作系统管理着CPU、内存、文件系统、网络接口等资源，LLM正在演变为一个管理着不同"模态进程"和"工具进程"的中央调度器。它接收用户的自然语言指令，调度不同的模块（视觉理解、代码执行、网络搜索、图像生成）来完成任务，然后将结果整合为统一的输出。这个"LLM OS"的愿景代表了人工智能发展的前沿方向。

---

## 9.4 关联内容

Karpathy关于LLM的演讲不止这一个。本节介绍两个密切相关的视频，它们从不同角度深化了对LLM的理解。

### 9.4.1 "Intro to Large Language Models"（1小时）：LLM的通俗介绍

这是Karpathy在2023年11月发布的一个面向普通观众的入门演讲[^2^]。如果说"State of GPT"是为技术从业者准备的深度讲解，那么"Intro to LLM"则是为所有人准备的通俗科普。

视频的核心内容分为三大部分：

**第一部分：LLM是什么，如何工作**。Karpathy解释了LLM的基本工作原理——它是一个被训练来预测下一个token的神经网络，这个看似简单的目标却催生了惊人的通用能力。他将LLM的运行比喻为"梦境"：模型在训练时"阅读"了整个互联网，然后在生成文本时进入一种"联想式梦境"状态，根据前面看到的内容来预测接下来应该出现什么。

这里有一个重要的mental model：不要把LLM想象成一个"数据库"（它存储和检索事实），而要把它想象成一个"梦境生成器"（它根据统计规律生成连贯的文本）。这个视角能帮助你理解为什么LLM会幻觉、为什么它能创造性地组合知识、以及为什么它有时会犯错。

**第二部分：LLM的未来发展方向**。Karpathy讨论了几个关键趋势：Scaling Laws（继续扩大规模）、Tool Use（工具使用）、Multimodality（多模态）、System 1/2 Thinking（快思考与慢思考）、以及Self-Improvement（自我改进）。其中"System 1/2 Thinking"的概念尤其深刻——他借鉴了心理学家Daniel Kahneman的框架，指出当前LLM的生成类似于"System 1"（快速、直觉式的思考），未来的方向是让LLM具备"System 2"（缓慢、审慎、深度推理）的能力。

**第三部分：LLM安全**。这是一个越来越重要的议题。Karpathy讨论了Jailbreaks（越狱——通过精心设计的prompt绕过模型的安全约束）、Prompt Injection（提示注入——在不可信输入中嵌入恶意指令）、以及Data Poisoning（数据投毒——在训练数据中加入有害内容）等安全挑战。

这个视频的观看量超过360万，是理解LLM最好的入门资源之一。幻灯片可以在Google Drive上找到[^2^]。

### 9.4.2 "Deep Dive into LLMs like ChatGPT"（3.5小时）：训练细节的深入讲解

2025年2月发布的这个视频[^3^]，是Karpathy对LLM技术最全面的系统性讲解。长达3.5小时的内容覆盖了从tokenization到pretraining、从post-training到实际应用策略的完整技术栈。

视频的核心价值在于它提供了大量"mental model"来帮助理解LLM的行为。以下是几个关键洞察：

**Pretraining的本质**。Karpathy将预训练描述为"压缩"（compression）——模型在试图压缩整个互联网的过程中，被迫学到了关于世界的深刻表示。这个过程产生了一个**基础模型（base model）**，它不是一个助手，而是一个"互联网文本模拟器"。当你向一个base model提问时，它不会回答你，而是会模拟互联网上接下来可能出现的内容——可能是答案，也可能是另一个问题，或者是某个论坛的讨论。

**Post-training的转变**。从base model到assistant的转变发生在post-training阶段（包括SFT和RLHF）。Karpathy强调，这个过程的本质不是"注入新知识"，而是"行为塑形"——模型学习了一种新的交互模式：用户提出问题，助手给出回答。它并不改变模型的基础知识，只是改变了模型"做梦"的风格。

**Tokenization的重要性**。视频用大量篇幅讨论了tokenization对模型能力的影响。模型的"基本认知单元"是token，不是字符或词。这意味着模型处理文本的方式与人类直觉不同——它可能 struggles with simple spelling tasks（拼写任务），因为它的"字母"是token而非字符。理解这一点对于理解LLM的 strengths 和 limitations 至关重要。

**Jagged Intelligence（锯齿状智能）**。这是Karpathy提出的一个概念，描述LLM能力的不均匀性。模型在某些任务上表现出超人的能力（如创意写作、代码生成），在另一些看似简单的任务上却表现不佳（如精确的字符级操作、某些逻辑谜题）。这种不均匀性是因为模型的能力来源于训练数据的统计结构——它在训练数据中频繁出现的模式上学得很好，在罕见模式上则表现较差。

---

## 9.5 推荐资源

以下资源按照学习路径组织，涵盖了论文、视频、代码项目和博客文章。

### 9.5.1 核心论文与视频链接

**必读论文**

| 论文 | 作者/年份 | 链接 | 重点内容 |
|------|----------|------|----------|
| Training language models to follow instructions with human feedback (InstructGPT) | Ouyang et al., 2022 | [arXiv](https://arxiv.org/abs/2203.02155) | 首次系统阐述RLHF训练流程的论文，三阶段范式（SFT→RM→RL）的标准参考 |
| Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al., 2023 | [arXiv](https://arxiv.org/abs/2307.09288) | 详细描述了Llama 2的两阶段RLHF训练、Ghost Attention等技术细节 |
| Learning to Summarize from Human Feedback | Stiennon et al., 2020 | [arXiv](https://arxiv.org/abs/2009.01325) | Reward Modeling方法论的奠基性论文，展示了从人类排序数据训练reward model的完整流程 |
| GPT-4 Technical Report | OpenAI, 2023 | [arXiv](https://arxiv.org/abs/2303.08774) | GPT-4的能力评估和安全考量 |

**必看视频**

| 视频 | 时长 | 链接 | 适合人群 |
|------|------|------|----------|
| State of GPT (本课内容) | 42分钟 | [YouTube](https://www.youtube.com/watch?v=bZQun8Y4L2A) | 已理解GPT基础架构，想了解完整训练流程的学习者 |
| Intro to Large Language Models | 1小时 | [YouTube](https://www.youtube.com/watch?v=zjkBMFhNj_g) | LLM完全新手，想建立整体认知的学习者 |
| Deep Dive into LLMs like ChatGPT | 3.5小时 | [YouTube](https://www.youtube.com/watch?v=7xTGNNLPyMI) | 想深入理解LLM每个技术细节的学习者 |

**代码项目**

| 项目 | 链接 | 说明 |
|------|------|------|
| llama2.c | [GitHub](https://github.com/karpathy/llama2.c) | Karpathy写的Llama 2推理实现，约500行C代码，是理解LLM推理过程的绝佳资源 |
| llm.c | [GitHub](https://github.com/karpathy/llm.c) | GPT-2训练的C/CUDA实现，与build-nanogpt逻辑等价但用纯C重写，用于学习高效实现 |
| build-nanogpt | [GitHub](https://github.com/karpathy/build-nanogpt) | 第10课内容对应的代码仓库，完整复现GPT-2的训练流程 |

**博客文章**

- Andrej Karpathy博客主页 [^4^]：https://karpathy.ai/ — 包含多篇关于神经网络和深度学习的经典文章
- "The Unreasonable Effectiveness of RNNs" [^5^]：http://karpathy.github.io/2015/05/21/rnn-effectiveness/ — Karpathy最著名的博客文章之一，展示了RNN在序列建模上的惊人能力，虽然写于2015年，但其中的直觉对理解LLM依然有价值

---

## 课后练习

1. **理解RLHF的完整流程**：画出Pretraining→SFT→RM→RLHF四阶段的流程图，标注每个阶段的输入数据、训练目标和输出模型。思考：为什么RLHF不能用普通的监督学习替代？

2. **KL散度的直觉**：KL散度 $D_{KL}(P \| Q)$ 衡量的是"用分布Q来近似真实分布P时损失的信息量"。在RLHF中，我们用KL散度约束策略不要偏离SFT模型太远。请用你自己的语言解释：为什么去掉这个约束会导致模式坍塌？

3. **Scaling Laws的实践应用**：假设你有一个1B参数的模型在某个基准测试上达到了70分。根据Scaling Laws的幂律关系 $L \propto N^{-\alpha}$（取 $\alpha \approx 0.07$），估算10B和100B参数模型在同一基准上可能达到的分数。讨论这个估算的局限性。

4. **Emergence现象的思考**：列举三个你认为的LLM涌现能力（可以是本章讨论的，也可以是你自己观察到的）。对于每个能力，尝试解释：为什么这个能力需要模型达到一定规模后才出现？小模型"缺"的是什么？

5. **实现一个简化的RAG系统**：使用OpenAI的embedding API（或开源embedding模型如sentence-transformers）和一个向量数据库（如FAISS），为一个你感兴趣的领域（如机器学习论文、技术文档、小说）构建一个RAG问答系统。对比纯LLM回答和RAG增强回答在事实准确性上的差异。

6. **Tool Use的设计挑战**：设计一个Agent系统来解决一个具体的实际问题（如"帮我规划一次从北京到东京的5日游，预算5000元"）。列出你的Agent需要哪些工具，描述它可能的推理和行动链。思考：哪些环节最容易出错？如何提高可靠性？

7. **观察模型的"梦境"**：使用一个开源的base model（如Llama 2 7B base，非chat版本），给它一些prompts（如问题、不完整的句子），观察它的续写行为。然后使用对应的chat版本做同样的实验。对比两者的差异，体会SFT对模型行为的改变。

---

[^1^]: Karpathy, A. (2023). *State of GPT*. Microsoft Build 2023. https://www.youtube.com/watch?v=bZQun8Y4L2A

[^2^]: Karpathy, A. (2023). *Intro to Large Language Models*. https://www.youtube.com/watch?v=zjkBMFhNj_g

[^3^]: Karpathy, A. (2025). *Deep Dive into LLMs like ChatGPT*. https://www.youtube.com/watch?v=7xTGNNLPyMI

[^4^]: Karpathy, A. Personal blog. https://karpathy.ai/

[^5^]: Karpathy, A. (2015). *The Unreasonable Effectiveness of Recurrent Neural Networks*. http://karpathy.github.io/2015/05/21/rnn-effectiveness/



---

## 10. 第10课：Reproduce GPT-2 — 从零复现124M模型

经过了前面九课的学习，我们已经走过了漫长的旅程。从第1课中手写一个微小的反向传播网络，到第7课用PyTorch搭建一个完整的GPT语言模型，再到第8课理解GPU的并行计算能力，每一步都在为这一刻做准备。现在，是时候将这些碎片拼合成一幅完整的图景了：我们将从零开始复现OpenAI的GPT-2 small模型——124M参数的真实规模Transformer，在FineWeb大规模数据集上进行训练。

Karpathy在4小时的视频中完整记录了整个过程。视频最终产出的build-nanogpt仓库与nanoGPT有约90%的相似度，但它从零开始构建，每一步都清晰可见。这不仅是代码的堆砌，更是一次深度学习工程实践的全景展示。

### 10.1 项目概览与目标

#### 10.1.1 目标与意义

复现GPT-2 small（124M参数）的目标是什么？不是为了超越它——GPT-2在2024年已经是一个"古老"的模型了。目标在于**理解**：通过亲手搭建每一个组件、配置每一个超参数、观察每一次训练迭代，我们真正理解了大语言模型训练的全貌。正如Karpathy所说，这个过程让我们从一个只会调用API的用户，变成一个懂得模型如何"呼吸"的工程师。

具体来说，我们要做到以下几点：

1. **搭建完整的GPT-2架构**：包括CausalSelfAttention、MLP、Block和完整的GPT类，然后加载Hugging Face的预训练权重来验证实现正确性。
2. **优化训练效率**：从原始的CPU baseline约1000ms每步，通过Tensor Cores、混合精度、torch.compile、Flash Attention和vocab size对齐，逐步优化到约93ms每步——超过10倍的加速。
3. **配置正确的训练流程**：按照GPT-2和GPT-3论文设置超参数，实现分布式训练、梯度累积、学习率调度等完整训练栈。
4. **在大规模真实数据上训练**：使用FineWeb EDU数据集（约100亿token），运行完整的预训练流程。
5. **评估模型能力**：通过验证集损失、HellaSwag常识推理评估和文本生成质量来判断训练效果。

#### 10.1.2 项目结构

最终的项目结构清晰而简洁：

```
build-nanogpt/
├── train_gpt2.py      # 主训练脚本（约520行）
├── play.ipynb         # Jupyter Notebook，用于可视化loss曲线等
├── fineweb.py         # FineWeb数据集下载与处理
├── hellaswag.py       # HellaSwag评估数据集处理
└── input.txt          # tiny Shakespeare示例数据
```

所有代码集中在`train_gpt2.py`一个文件中，这体现了Karpathy的设计哲学：**简单、可读、教育性优先**。对于生产环境，代码需要更多工程化的封装，但对于学习而言，将一切放在眼前是最有效的方式。

#### 10.1.3 与之前课程的联系

这一课本质上是对前面课程的一次"压力测试"。第7课的玩具GPT和第10课的真实GPT-2在结构上是相同的——都是Transformer decoder-only架构。区别只在于规模：从第7课的数百万参数扩展到124M参数，从KB级的 Shakespeare文本扩展到10B token的FineWeb数据集，从单CPU/GPU训练扩展到多GPU分布式训练。如果你理解了第7课的架构，你已经理解了GPT-2的80%；剩下的20%是工程优化和训练技巧。

### 10.2 GPT-2架构详解

让我们从最核心的部分开始：GPT-2的神经网络架构。好消息是，如果你跟上了第7课，这里的结构会非常熟悉。GPT-2本质上就是一个更大的、配置更精细的GPT模型。

#### 10.2.1 GPTConfig：配置即架构

Karpathy首先用一个dataclass来定义模型的所有配置参数。这种方式非常优雅——**配置即架构**，改变配置就能实例化不同规模的模型：

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大序列长度（上下文窗口）
    vocab_size: int = 50257  # 词表大小：50,000个BPE merges + 256字节token + 1个<|endoftext|>
    n_layer: int = 12        # Transformer层数
    n_head: int = 12         # 注意力头数
    n_embd: int = 768        # 嵌入维度
```

通过修改这些参数，我们就能得到GPT-2家族的不同成员。GPT-2 small（124M）使用上述默认值；medium（350M）将`n_layer`设为24、`n_head`设为16、`n_embd`设为1024；large（774M）和xl（1558M）继续扩大。这种**缩放定律（scaling law）**——增大模型规模就能提升性能——正是大语言模型研究的核心发现之一。

#### 10.2.2 CausalSelfAttention：自注意力的核心

自注意力层是整个Transformer的灵魂。GPT-2的实现有一个值得注意的技巧：用一个线性层同时计算Q、K、V，而不是分别定义三个独立的线性层。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 关键技巧：单个线性层同时计算Q, K, V，然后split
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 标记这是残差路径的输出层，需要特殊初始化缩放
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        # 计算QKV并split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape为(B, n_head, T, head_size)以支持多头并行计算
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash Attention（PyTorch内置，自动使用最优kernel）
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 合并多头输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

让我们逐段解析这段代码。首先，`c_attn`（combined attention）是一个从`n_embd`到`3 * n_embd`的线性变换。为什么要合并Q、K、V的计算？这主要是出于效率考虑——一个矩阵乘法操作通常比三个独立的矩阵乘法更快，因为可以更好地利用GPU的并行计算能力。在`forward`中，我们用`split`将输出重新分成三个张量。

接下来是reshape和transpose操作。关键直觉是：注意力计算在每个头（head）内部独立进行。我们有`n_head`个头，每个头处理的维度是`C // n_head`（768 / 12 = 64）。通过transpose，我们将张量形状从`(B, T, n_head, head_size)`变为`(B, n_head, T, head_size)`，这样每个头的注意力计算就可以在GPU上并行执行。

最后，也是最重要的：`F.scaled_dot_product_attention` 是PyTorch 2.0引入的内置函数，它自动使用**Flash Attention**算法。Flash Attention是一种IO-aware的精确注意力实现，通过分块（tiling）和重计算（recomputation）来减少对高带宽显存（HBM）的访问，从而在不牺牲精度的情况下大幅提升速度和降低显存占用。我们不需要手动实现因果掩码——只需要传入`is_causal=True`，函数会自动确保每个位置只能attend到它自己和前面的位置。

#### 10.2.3 MLP层：GELU激活与前馈网络

每个Transformer Block除了注意力层外，还包含一个两层的全连接前馈网络（MLP）：

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # c_fc: "fully connected"，将维度扩大4倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU激活函数（使用tanh近似，与原始GPT-2一致）
        self.gelu = nn.GELU(approximate='tanh')
        # c_proj: 投影回原始维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```

MLP的结构非常简单：先将维度从768扩展到3072（4倍），经过GELU激活函数，再投影回768。为什么要扩大4倍？这是原始Transformer论文的设计选择，后来的研究发现这个比例可以在参数量和表达能力之间取得良好平衡。

这里使用的激活函数是**GELU（Gaussian Error Linear Unit）**而非ReLU。GELU的定义是：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中$\Phi(x)$是标准正态分布的累积分布函数。直观上，GELU可以被理解为一种"平滑的ReLU"——它不像ReLU那样在零点硬切换，而是以概率的方式平滑过渡。GELU在Transformer架构中表现优于ReLU，原因可能在于它的平滑性质使得梯度流动更稳定。GPT-2原始实现使用了tanh近似版本，我们遵循这一选择以保持数值一致性。

#### 10.2.4 Block结构：Pre-Norm vs Post-Norm

有了注意力层和MLP层，现在把它们组合成一个Transformer Block。GPT-2使用**Pre-LayerNorm（Pre-LN）**结构，这是与原始Transformer论文的一个重要区别：

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN: 先LayerNorm，再执行子层，最后加残差
        x = x + self.attn(self.ln_1(x))   # 残差连接 + Attention
        x = x + self.mlp(self.ln_2(x))    # 残差连接 + MLP
        return x
```

Pre-LN与Post-LN的区别至关重要。原始Transformer论文使用的是Post-LN：先执行子层（Attention/MLP），再加残差，最后LayerNorm。而Pre-LN将LayerNorm移到子层之前。为什么要这样做？

核心问题是训练稳定性。在Post-LN结构中，LayerNorm位于残差路径上，会改变梯度在残差连接中的流动方式。当模型层数较深时，梯度需要通过多个LayerNorm层回传，容易导致梯度消失或爆炸。Pre-LN将LayerNorm移出残差路径，使得梯度可以更直接地通过残差连接流动，从而大幅提升深层模型的训练稳定性。GPT-2有12层，GPT-3有96层——如果没有Pre-LN，训练如此深的模型会非常困难。

#### 10.2.5 完整的GPT类

现在将所有组件组装成完整的GPT模型：

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  # 最终LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重共享：wte和lm_head共享同一个权重矩阵
        self.transformer.wte.weight = self.lm_head.weight
        # 应用自定义初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # 对残差路径的输出层进行缩放初始化
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

这个结构与我们在第7课中搭建的模型几乎完全一致，但有几个关键的工程细节值得深入讨论。

**权重共享（Weight Tying）** 是第一项重要技术。注意这一行：`self.transformer.wte.weight = self.lm_head.weight`。`wte`（word token embedding）将token ID映射到嵌入向量，`lm_head`将最终的嵌入向量映射回词表上的概率分布。令这两个矩阵共享权重是一种参数效率极高的设计——它将嵌入层的参数从训练中"回收"用于输出层。直觉上，输入嵌入和输出投影做的是互逆的操作：一个将离散token编码为连续向量，另一个将连续向量解码为token概率。共享权重让模型在这两个任务上学到的知识相互促进。

**定制化权重初始化** 是第二项关键技术。Transformer的训练对初始化非常敏感。这里我们使用标准差0.02的正态分布初始化所有Linear和Embedding层。但对于残差路径的输出层（`c_proj`），我们应用了额外的缩放：`std *= (2 * n_layer) ^ -0.5`。为什么要这样做？

这个缩放的数学依据来自Goyal et al.的研究（也在GPT-2论文中被提及）。在一个有残差连接的深层网络中，每层的输出通过加法累积到主路径上。如果我们有$n$层，每层残差路径的输出方差都相似，那么经过所有层之后，累积的方差会大约是单层的$2n$倍（因为每个Block有两条残差路径：attention和MLP）。为了让最终输出的方差保持在合理范围内，我们需要将每条残差路径输出层的初始化标准差除以$\sqrt{2n}$。这种**残差缩放初始化**是深层Transformer稳定训练的关键之一。

```python
    def forward(self, idx, targets=None):
        B, T = idx.size()
        # Token嵌入 + 位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # 通过所有Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # 计算损失（如果提供了targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

`forward`方法同样熟悉：token嵌入加上位置嵌入，依次通过所有Transformer blocks，最终的LayerNorm，然后通过`lm_head`生成每个位置对所有可能token的logits。如果提供了target tokens，就计算交叉熵损失——本质上是在做下一个token预测。

#### 10.2.6 加载预训练权重验证

搭建好架构后，Karpathy做的第一件事就是从Hugging Face加载官方GPT-2的预训练权重来验证实现正确性。这是一个至关重要的**完整性检查（sanity check）**：如果我们的模型架构有任何细微偏差，加载权重后输出的数值就会与官方不同。

```python
    @classmethod
    def from_pretrained(cls, model_type):
        """从HuggingFace加载预训练GPT-2权重来验证实现"""
        from transformers import GPT2LMHeadModel
        # 创建配置
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config = GPTConfig(**config_args)
        model = GPT(config)
        # 加载HuggingFace权重并映射到我们的模型
        # ...（权重名称映射和加载逻辑）
        return model
```

验证的过程非常直观：给模型输入一组token，比较我们模型的输出logits与Hugging Face官方模型的输出。如果两者完全一致（在浮点精度允许的误差范围内），就说明我们的架构实现是正确的。这种**增量式验证**的工程实践值得学习：每完成一个组件，就用已知的正确答案来验证它，而不是等到所有代码写完才发现问题。

### 10.3 训练优化技术

架构验证通过后，真正的挑战开始了：如何让训练既快又稳定？Karpathy在视频中展示了从约1000ms每步优化到约93ms每步的完整过程，并配合了分布式训练、混合精度、梯度累积等技术。这一节是整个项目工程实践的核心。

#### 10.3.1 分布式训练：DDP

当单张GPU的显存或算力不足以支撑训练时，我们需要**分布式数据并行（Distributed Data Parallel, DDP）**。DDP的核心思想非常简单：每张GPU处理不同的数据子集，各自计算梯度，然后在每次反向传播后同步（取平均）梯度。这样，所有GPU上的模型参数始终保持一致，但吞吐量随GPU数量线性增长。

```python
# 检测是否处于DDP环境
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')  # NCCL是NVIDIA的GPU通信库
    ddp_rank = int(os.environ['RANK'])          # 全局进程ID
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本地GPU ID
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总GPU数
    device = f'cuda:{ddp_local_rank}'
    master_process = (ddp_rank == 0)  # 只有rank 0进程负责打印和保存
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True

# 包装模型
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # 原始模型（解包装）
```

DDP通过`torchrun`命令启动：`torchrun --standalone --nproc_per_node=8 train_gpt2.py`。`torchrun`会自动设置`RANK`、`LOCAL_RANK`、`WORLD_SIZE`等环境变量。NCCL（NVIDIA Collective Communications Library）是底层的GPU间通信库，它通过PCIe或NVLink高效地在GPU之间传输梯度数据。

DDP的一个关键优化是**梯度同步的时机控制**。在梯度累积的场景下，我们只在最后一个micro-step才需要同步梯度：

```python
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    # 只在最后一次backward时同步梯度
    if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    # 前向和反向传播
    logits, loss = model(x, y)
    loss.backward()
```

这行`require_backward_grad_sync`是一个重要的性能优化——如果我们每一步都同步梯度，会产生大量不必要的通信开销。

#### 10.3.2 混合精度训练：bfloat16与TF32

神经网络训练本质上对数值精度并不那么敏感。32位浮点数（float32）的精度对于权重更新来说是一种浪费。如果我们能用16位来计算，就能在相同时间内处理更多数据，同时减少显存占用。

但直接使用16位浮点数（float16）有一个致命问题：它的指数范围太小（5位），容易导致梯度下溢（underflow）——非常小的梯度变成零，模型就停止学习了。**bfloat16**解决了这个问题：

| 格式 | 指数位 | 尾数位 | 总位数 | 特点 |
|------|--------|--------|--------|------|
| float32 | 8 | 23 | 32 | 标准精度 |
| float16 | 5 | 10 | 16 | 范围小，易underflow |
| bfloat16 | 8 | 7 | 16 | 保持float32的范围，精度略降 |

bfloat16保留了float32的8位指数（所以数值范围与float32相同），只减少尾数到7位。这意味着它几乎不会出现float16那样的underflow问题，同时享受16位运算的速度优势。在现代NVIDIA GPU（A100及以上）上，bfloat16运算通过Tensor Cores加速，速度可达float32的数倍。

在PyTorch中启用混合精度训练非常简单：

```python
# 设置TF32精度（在使用float32时自动使用Tensor Cores）
torch.set_float32_matmul_precision('high')

# 在训练循环中使用autocast自动管理精度
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

`torch.autocast`是一个上下文管理器，它会自动将符合条件的操作转换为bfloat16，同时在需要的地方保持float32（比如Softmax和损失计算）。开发者不需要手动管理哪些操作用哪种精度。

#### 10.3.3 梯度累积：模拟大批量训练

大批量训练通常更稳定，也能更好地利用GPU的并行计算能力。但大批量意味着更大的显存占用——单张GPU可能装不下。梯度累积通过在多个小批次上分别计算梯度、累加起来、最后统一更新参数，来模拟大批次训练的效果。

```python
# 目标总批次大小：约0.5M tokens
total_batch_size = 524288  # 2^19
B = 64      # micro batch size（每步处理的样本数）
T = 1024    # 序列长度
# 计算梯度累积步数
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# 训练循环中使用梯度累积
optimizer.zero_grad()
loss_accum = 0.0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    # 关键：损失必须按累积步数缩放
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()  # 梯度自动累加
optimizer.step()  # 统一更新参数
```

注意这一行关键代码：`loss = loss / grad_accum_steps`。为什么必须缩放损失？因为PyTorch的梯度是损失对参数的导数。如果不缩放，在`grad_accum_steps`个micro-step上分别backward，累积的梯度相当于总损失的`grad_accum_steps`倍。将每个micro-step的损失除以累积步数后，最终累积的梯度就等价于一个大批次的梯度。

#### 10.3.4 学习率调度：warmup + cosine decay

训练一个124M参数的模型，学习率的选择直接决定训练能否成功。GPT-2论文和GPT-3论文都使用了相同的调度策略：**先warmup，再cosine decay**。

```python
max_lr = 6e-4
min_lr = max_lr * 0.1    # 最低学习率为最大值的10%
warmup_steps = 715       # 前715步线性增加
max_steps = 19073        # 总训练步数

def get_lr(it):
    # 阶段1：线性warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 阶段3：超过max_steps后保持min_lr
    if it > max_steps:
        return min_lr
    # 阶段2：cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

为什么要warmup？训练初期，模型参数是随机初始化的，梯度方向不稳定。如果一开始就用大学习率，参数可能在损失函数的高维空间中"跳跃"到糟糕的区域，导致训练发散。warmup通过在前几百步线性增加学习率，让模型有一个"热身"的过程，逐步适应梯度的方向。

cosine decay的直觉则更为优雅：它让学习率像余弦曲线一样平滑下降。与阶梯式衰减（step decay）相比，cosine decay在后期使用极小的学习率进行精细调整，帮助模型收敛到损失函数的更平坦区域——这通常对应更好的泛化性能。

warmup步数715的取值也有讲究。GPT-2论文使用375M tokens的warmup，batch size为524,288 tokens，所以warmup steps = 375M / 524K ≈ 715。这是一个需要根据总训练量和batch size来调整的超参数。

#### 10.3.5 优化器：AdamW与权重衰减

AdamW是训练Transformer的标准优化器。与原始Adam相比，AdamW实现了**解耦的权重衰减（decoupled weight decay）**——权重衰减不再通过梯度实现，而是直接在参数更新时进行。这虽然是一个细微的数学区别，但在实践中对训练稳定性有显著影响。

GPT-2使用以下AdamW超参数：
- 学习率：6e-4
- betas：(0.9, 0.95) —— beta2比默认的0.999略小，意味着对梯度二阶矩的EMA更"敏感"
- epsilon：1e-8
- 权重衰减：0.1

但权重衰减并不应用于所有参数。一个重要的工程技巧是**按参数维度分组应用权重衰减**：

```python
def configure_optimizers(self, weight_decay, learning_rate, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    # ≥2维的参数（weight tensors）应用weight decay
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # <2维的参数（biases, layernorm参数）不应用weight decay
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # 使用fused AdamW（CUDA上的更快实现）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                   betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer
```

这个分组的逻辑是什么？权重衰减是一种L2正则化，它惩罚大权重值，鼓励模型使用更简单的权重分布。但biases和LayerNorm的参数尺度对训练稳定性很关键——对它们应用权重衰减反而会干扰训练。具体来说，biases通常初始化为零，LayerNorm的参数初始化为1（gamma）和0（beta），这些值不需要正则化。只有二维及以上的weight tensors（如Linear层的权重矩阵）才从权重衰减中受益。

此外，`fused=True`启用PyTorch的融合AdamW实现，它将多个kernel操作融合为一个，减少GPU kernel launch的开销。

#### 10.3.6 torch.compile与Flash Attention：速度飞跃

在完成上述训练配置后，Karpathy还进行了一系列性能优化，将每步时间从约1000ms降到了约93ms。这些优化虽然不直接影响训练质量，但决定了训练在工程上是否可行。

```python
# torch.compile：PyTorch 2.0的JIT编译器
model = torch.compile(model)
```

`torch.compile`是PyTorch 2.0引入的革命性功能。它通过图捕获将Python代码编译为优化的CUDA kernel，消除了大量的Python解释器开销，并实现了**kernel fusion**——将多个小操作合并为一个大操作，减少内存读写。在这个项目中，torch.compile alone就将每步时间从300ms降到130ms。

Flash Attention（通过`F.scaled_dot_product_attention`）则贡献了另一大块加速：从130ms降到96ms。它通过重新组织attention计算的内存访问模式，大幅减少了对高带宽显存的读写。

最后一步优化是**vocab size对齐**：将词表从50257填充到50304（64的倍数）。这看似微不足道，但GPU kernel在处理"整齐"的维度时效率更高。这一改动将96ms进一步优化到93ms。

| 优化步骤 | 每步时间 | 关键技术 |
|----------|----------|----------|
| CPU baseline | ~1000ms | 无 |
| Tensor Cores + TF32 | 333ms | GPU Tensor Core, TF32混合精度 |
| bfloat16 + 梯度缩放 | 300ms | bfloat16数据类型 |
| torch.compile | 130ms | JIT编译, kernel fusion |
| Flash Attention | 96ms | IO-aware attention算法 |
| vocab size对齐(50257→50304) | 93ms | 对齐到64的倍数 |

从1000ms到93ms，超过10倍的加速。这体现了深度学习工程的一个核心真理：**好的实现比好的算法更重要**。一个理论上正确的训练流程，如果工程实现低效，可能在实践中完全不可行。

### 10.4 数据管道

模型和优化器准备好后，下一个关键组件是数据管道。数据的质量和数量直接决定了模型的上限。本节我们讨论FineWeb数据集和数据加载的实现。

#### 10.4.1 FineWeb数据集

GPT-2使用WebText数据集进行训练——从Reddit上获得至少3个 Karma 的外部链接中抓取的网页文本，约40GB。GPT-3则大幅扩展了数据来源，包括Common Crawl的过滤版本、WebText2、Books和Wikipedia，总计约300B token。

在本项目中，Karpathy选择了**FineWeb EDU**数据集。FineWeb是对Common Crawl进行大规模清洗和去重后得到的高质量网页文本数据集，而FineWeb EDU进一步筛选出了教育领域的内容。选择这个数据集的理由是：教育内容通常质量更高、语言更规范，模型在这种数据上训练可以学到更好的语言表达和推理能力。数据总量约100亿token（10B），这与GPT-2的训练规模（约10B token）相当。

#### 10.4.2 DataLoader实现

数据加载器的设计需要支持两个关键需求：**分布式加载**（每个GPU进程加载不同的数据段）和**高效读取**（按shard加载，避免一次性将全部数据载入内存）。

```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B  # batch size
        self.T = T  # sequence长度
        self.process_rank = process_rank      # 当前进程ID
        self.num_processes = num_processes    # 总进程数
        data_root = "edu_fineweb10B"
        shards = sorted([s for s in os.listdir(data_root) if split in s])
        self.shards = [os.path.join(data_root, s) for s in shards]
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # 每个进程从不同的位置开始读取
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position + self.B*self.T + 1]
        x = (buf[:-1]).view(self.B, self.T)   # inputs
        y = (buf[1:]).view(self.B, self.T)    # targets（右移1位）
        # 所有进程一起前进
        self.current_position += self.B * self.T * self.num_processes
        # 如果当前shard读完了，切换到下一个
        if (self.current_position + self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y
```

`DataLoaderLite`的设计体现了几个重要的工程考量。首先是**进程间的数据切分**：每个进程的起始位置不同，进程`i`从位置`B * T * i`开始。读取数据时，所有进程一起前进`B * T * num_processes`个token，确保不会有数据重叠或遗漏。其次是**shard管理**：数据被分成多个文件（shard），读完一个shard后自动切换到下一个，并在所有shard间循环。这种设计允许我们处理比内存大得多的数据集——任何时候只有当前shard被加载到内存中。`load_tokens`函数通常从磁盘读取预token化的numpy数组，直接在内存中操作，避免了训练时实时tokenize的开销。

### 10.5 评估与调试

训练一个大模型是一项昂贵的投资——在A100上运行约19,000步大约需要一天时间。在训练过程中，我们需要持续监控模型状态，确保训练在正常进行，并在训练完成后评估模型的实际能力。

#### 10.5.1 训练监控：loss曲线与关键指标

训练循环中，Karpathy在每一步打印以下指标：

```python
torch.cuda.synchronize()  # 等待GPU完成所有操作
dt = time.time() - t0     # 本步实际耗时
tokens_per_sec = (train_loader.B * train_loader.T *
                  grad_accum_steps * ddp_world_size / dt)
if master_process:
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | "
          f"lr {lr:.4e} | norm: {norm:.4f} | "
          f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
```

这些指标的含义如下：

- **loss**：训练损失。一个健康训练的标志是loss稳步下降。GPT-2 124M在FineWeb上训练的最终训练loss大约在2.8-3.0左右。
- **lr（学习率）**：当前学习率。应该按照warmup + cosine decay的曲线变化。
- **norm（梯度范数）**：梯度裁剪前的梯度L2范数。如果这个值突然变得非常大（比如超过10），说明训练可能不稳定。
- **dt**：每步耗时，用于监控训练效率。
- **tok/sec**：每秒处理的token数，衡量整体吞吐量。

每250步，还会进行额外的评估：

```python
# 验证集损失
if step % 250 == 0:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(20):
            x, y = val_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += loss / 20
```

验证损失（validation loss）是评估过拟合的关键指标。如果训练loss持续下降但验证loss停滞或上升，说明模型在过拟合训练数据。在本项目中，由于训练数据量（10B tokens）远大于模型容量（124M参数），过拟合的风险相对较低——模型甚至还没"记住"所有数据，训练就结束了。

#### 10.5.2 HellaSwag评估：常识推理的零样本测试

Loss数字告诉我们模型在预测下一个token上表现如何，但它不能直接衡量模型的"理解能力"。**HellaSwag**是一个常识推理评估基准，用于测试模型的零样本（zero-shot）推理能力。

HellaSwag的格式是这样的：给定一个场景的描述（比如"一个女人在厨房里做蛋糕"），模型需要从四个可能的后续事件中选出最合理的一个。正确的后续应该符合现实世界的常识（比如"她把蛋糕放进烤箱"），而干扰项则看似合理但实际上不可能（比如"她把蛋糕放进洗衣机"）。

评估方法是**零样本**的——模型没有针对HellaSwag进行过任何微调。对于每个问题，我们将四个选项分别拼接到前缀后面，计算模型对完整序列的perplexity（即交叉熵损失的指数），选择perplexity最低的那个选项作为预测答案。

```python
# HellaSwag评估的核心逻辑
# 对每个问题的四个选项，计算模型的perplexity
def eval_hellaswag(model):
    # ... 加载HellaSwag数据集
    correct = 0
    total = 0
    for example in hellaswag_data:
        prefix = example['context']     # 场景描述
        endings = example['endings']     # 四个选项
        # 对每个选项计算log probability
        logits, _ = model(tokens_with_ending)
        # 选择log prob最高的选项
        pred = argmax(log_probs)
        if pred == example['label']:
            correct += 1
        total += 1
    return correct / total  # 准确率
```

GPT-2 124M在HellaSwag上的准确率通常在25-30%左右（随机猜测是25%），而经过充分训练的更大模型可以达到50%以上。这个评估指标虽然简单，但能很好地反映模型对常识性因果关系的理解程度。

#### 10.5.3 文本生成：直观的质量检查

除了量化指标，定期从模型生成文本也是判断训练质量的直观方法：

```python
if step > 0 and step % 250 == 0:
    model.eval()
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    # 自回归生成
    for _ in range(50):
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)
    # 解码并打印
    generated = enc.decode(tokens.squeeze(0).tolist())
    print(generated)
```

这里使用了**top-k采样**：在每个生成步骤中，只考虑概率最高的50个token，然后按它们的相对概率进行采样。这平衡了生成质量（过滤掉极低概率的无意义token）和多样性（不完全贪心选择最高概率token）。随着训练的进行，生成的文本应该逐渐从胡言乱语变成语法通顺、语义连贯的段落。

### 10.6 课后练习

1. **架构验证练习**：实现`from_pretrained`方法的完整权重映射逻辑。加载Hugging Face的GPT-2权重，确保你的模型与官方模型在相同输入下产生完全一致的logits。

2. **注意力可视化**：在CausalSelfAttention中保存注意力权重矩阵，选择一段输入文本，可视化不同注意力头在关注哪些位置。你能发现哪些头倾向于关注附近的token，哪些头关注远距离的依赖？

3. **学习率调度实验**：修改warmup步数和max_steps，观察不同调度策略对训练稳定性和最终loss的影响。尝试使用线性衰减替代cosine decay，比较两者的差异。

4. **梯度累积实现**：在不使用PyTorch自动梯度累积的情况下，手动实现梯度累积的逻辑。你需要直接操作`param.grad`来累加梯度，并在更新后手动清零。

5. **DDP单进程模拟**：即使只有一块GPU，也可以模拟DDP的训练流程。使用`torchrun --standalone --nproc_per_node=1 train_gpt2.py`运行训练脚本，确保代码在非分布式环境下也能正确工作。

6. **性能分析**：使用PyTorch Profiler（`torch.profiler`）分析训练循环，找出耗时最长的操作。尝试关闭torch.compile或Flash Attention中的某一个，量化它们各自的加速贡献。

7. **更大模型的探索**：将配置改为GPT-2 medium（350M参数）或large（774M参数），在单张A100上能否装下？需要调整哪些超参数（如batch size、梯度累积步数）？

8. **数据集对比**：将FineWeb EDU替换为tiny Shakespeare数据集（从第7课沿用），在相同训练配置下比较训练速度和最终loss。为什么在小数据集上训练需要更小的max_steps？

### 10.7 推荐学习资源

**核心代码仓库**

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — 本课配套代码仓库，Git commits被精心设计为逐步构建过程，可以跟随commit history逐步重建整个项目[^1^]。
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy的生产级GPT训练框架，功能更完善但代码同样保持高度可读[^2^]。
- [llm.c](https://github.com/karpathy/llm.c) — 用纯C/CUDA实现的等效训练流程，速度更快，展示了如何将PyTorch代码翻译到更底层的实现[^3^]。

**论文与学术资源**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. Transformer架构的奠基论文，首次提出了自注意力和多头注意力机制[^4^]。
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019. GPT-2论文，详细描述了模型架构、训练过程和涌现能力[^5^]。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Brown et al., 2020. GPT-3论文，展示了更大模型的few-shot学习能力，并公布了完整的训练超参数[^6^]。

**生产级替代实现**

- [litGPT](https://github.com/Lightning-AI/litgpt) — Lightning AI维护的预训练LLM实现，支持多种架构（GPT-2、Llama、Falcon等），代码高度优化且文档完善[^7^]。
- [TinyLlama](https://github.com/jzhang38/TinyLlama) — 一个紧凑高效的Llama架构实现，展示了如何在有限资源上训练高性能的小模型[^8^]。

**Karpathy的相关视频**

- [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 第7课视频，从零搭建GPT模型的基础，是本章的直接前置知识[^9^]。
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 深入理解BPE分词器的工作原理，解释了GPT-2词表大小50257的由来[^10^]。
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — 2025年发布的3.5小时深度解析，覆盖LLM的完整训练栈，是对本课内容的极好补充[^11^]。

**社区与工具**

- [Lambda GPU Cloud](https://lambdalabs.com) — 提供按小时计费的GPU云服务，适合运行本项目的训练[^12^]。
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — 工业标准的预训练模型库，包含GPT-2的官方实现和预训练权重[^13^]。

---

至此，我们从零开始复现了GPT-2 124M模型的完整流程——从搭建架构、优化训练效率，到配置分布式训练、构建数据管道，再到评估模型能力。这个过程浓缩了深度学习中最重要的工程实践：架构设计、训练稳定性、性能优化和评估方法论。

但更重要的，是这十节课教会我们的思维方式。从第1课到第10课，我们从理解一个神经元的工作原理，一步步走到训练一个有1.24亿参数的Transformer模型。这个旅程的核心信念是：**深度学习的复杂性不在于任何单一概念的难度，而在于大量简单概念的精妙组合**。当你能够拆开每一个组件、理解每一个细节、亲手搭建每一个部分时，这个曾经看似神秘的领域就变成了可以理解和掌握的技术。

大语言模型的研究仍在飞速演进。GPT-2在2019年是一个突破，GPT-3在2020年震撼了世界，GPT-4在2023年将能力边界推向了新的高度。但不管模型如何变大、技术如何迭代，底层的原理始终相通：神经网络通过梯度下降学习数据的规律，Transformer通过注意力机制建模序列的依赖，而好的工程实践决定了这些理论能否转化为有效的训练。

愿你在自己的深度学习旅程中，保持好奇心、耐心和动手实践的热情。

