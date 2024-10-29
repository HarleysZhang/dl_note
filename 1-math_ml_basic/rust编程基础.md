## 认识 Cargo

`cargo` 是 `Rust` 的包管理工具，提供了从项目的建立、构建到测试、运行乃至部署的完整功能，与 `Rust` 语言及其编译器 `rustc` 紧密结合。

1，创建项目。

```bash
$ cargo new world_hello
$ cd world_hello
```

创建的项目结构：

```console
$ tree
.
├── .git
├── .gitignore
├── Cargo.toml
└── src
    └── main.rs
```

2，两种方式可以运行项目：

- `cargo run release`（默认是 debug），相当于执行了两个命令 cargo build 编译项目，和 ./target/debug/world_hello。
- 手动编译和运行项目。

3，`Cargo.toml` 和 `Cargo.lock`

-  `Cargo.toml` 是一种轻量级的配置文件格式，用于配置项目的元信息、依赖关系、构建选项等。
- `Cargo.lock` 文件是 `cargo` 工具根据同一项目的 `toml` 文件生成的项目依赖详细清单，因此我们一般不用修改它。

## 一，宏

1， `Package` 是一个项目工程，而包只是一个编译单元，`src/main.rs` 和 `src/lib.rs` 都是编译单元，因此它们都是包。

2， 一个 `crate` 可以是二进制（`src/main.rs`）或者库（`src/lib.rs`），每一个包（crate）都有包跟（ crate root），例如二进制包的包根是 `src/main.rs`，库包的包根是 `src/lib.rs`。它是编译器开始处理源代码文件的地方，同时也是包模块树的根部。

3，`rust` 出于安全考虑，默认情况下，所有的类型都是私有化的，包括函数、方法、结构体、枚举、常量，是的，就连模块本身也是私有化的。在 Rust 中，**父模块完全无法访问子模块中的私有项**，反之，子模块可以访问父模块。

4，模块可见性不代表模块内部项的可见性，模块的可见性仅仅是允许其它模块去引用它，但是想要引用它内部的项，还得继续将对应的项标记为 `pub`。

5，使用 `super` 引用模块，`super` 代表的是父模块为开始的引用方式。

```rust
fn serve_order() {}

// 厨房模块
mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order(); // 调用了父模块(包根)中的 serve_order 函数
    }

    fn cook_order() {}
}
```

### 1.1，属性

可以使用称被为宏的自定义句法形式来扩展 Rust 的功能和句法。宏需要被命名，并通过一致的句法去调用：`some_extension!(...)`。

定义新宏有两种方式：

- [声明宏(Macros by Example)](https://rustwiki.org/zh-CN/reference/macros-by-example.html)以更高级别的声明性的方式定义了一套新句法规则。
- [过程宏(Procedural Macros)](https://rustwiki.org/zh-CN/reference/procedural-macros.html)可用于实现自定义派生。

#### 1.1.1，过程宏

*过程宏*允许在执行函数时创建句法扩展。过程宏有三种形式:

- [类函数宏(function-like macros)](https://rustwiki.org/zh-CN/reference/procedural-macros.html#function-like-procedural-macros) - `custom!(...)`
- [派生宏(derive macros)](https://rustwiki.org/zh-CN/reference/procedural-macros.html#derive-macros)- `#[derive(CustomDerive)]`
- [属性宏(attribute macros)](https://rustwiki.org/zh-CN/reference/procedural-macros.html#attribute-macros) - `#[CustomAttribute]`

1，**派生宏**
派生宏为派生(derive)属性定义新输入。这类宏在给定输入结构体(struct)、枚举(enum)或联合体(union) token流的情况下创建新程序项。它们也可以定义派生宏辅助属性。

2，**属性宏**
属性宏定义可以附加到程序项上的新的外部属性，这些程序项包括外部(extern)块、固有实现、trate实现，以及 trait声明中的各类程序项。

### 1.2，使用 tracing 记录日志

1，**在于日志只能针对某个时间点进行记录，缺乏上下文信息，而线程间的执行顺序又是不确定的，因此日志就有些无能为力**。而 `tracing` 为了解决这个问题，引入了 `span` 的概念( 这个概念也来自于分布式追踪 )，一个 `span` 代表了一个时间段，拥有开始和结束时间，在此期间的所有类型数据、结构化数据、文本数据都可以记录其中。

2，tracing` 中最重要的三个概念是 `  `Span` 、` Event ` 和 `Collector`。

#### 1.2.1，使用方法-span! 宏

`span!` 宏可以用于创建一个 `Span` 结构体，然后通过调用结构体的 `enter` 方法来开始，再通过超出作用域时的 `drop` 来结束。

#### 1.2.2，使用方法-#[instrument]

如果想要将某个函数的整个函数体都设置为 `span` 的范围，最简单的方法就是为函数标记上 `#[instrument]`，此时 tracing 会自动为函数创建一个 span，span 名跟函数名相同，在输出的信息中还会自动带上函数参数。

## 二，模式匹配

match分支匹配的用法非常灵活，它的基本语法为：

```rust
match VALUE {
  PATTERN1 => EXPRESSION1,
  PATTERN2 => EXPRESSION2,
  PATTERN3 => EXPRESSION3,
}
```

`match` 匹配的通用形式：

```rust
match target {
    模式1 => 表达式1,
    模式2 => {
        语句1;
        语句2;
        表达式2
    },
    _ => 表达式3
}
```

**match** 支持两种匹配模式（不可反驳的模式(irrefutable) 和可反驳的的模式(refutable)）：

- 当明确给出分支的Pattern时，必须是可反驳模式，这些模式允许匹配失败
- 使用`_`作为最后一个分支时，是不可反驳模式，它一定会匹配成功
- 如果只有一个Pattern分支，则可以是不可反驳模式，也可以是可反驳模式

### 2.1，全模式列表

1，匹配字面值

2，匹配命名变量

3，单分支多模式

4，通过序列 ..= 匹配值的范围

5，解构并分解值： 使用模式来解构结构体、枚举、元组、数组和引用。

模式匹配一样要类型相同。

## 三，复合类型

### 3.1，结构体

结构体和元组类似，都是由多种类型组合而成。示例：

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
```

该结构体名称是 `User`，拥有 4 个字段，且每个字段都有对应的字段名及类型声明，例如 `username` 代表了用户名，是一个可变的 `String` 类型。

## 四，难点语法速记

1，Option 代表可能为空可能有值的一种类型，本质上是一个枚举，有两种分别是 Some 和 None。Some 代表有值，None 则类似于 null，代表无值。

2，Result 直接翻译过来就是“结果”，想象一下，我们的接口，服务有非常常规的调用场景，正常返回值，异常返回错误或抛异常等等。而 Rust 里就定义有了一个 Result 用于此场景。Result 内部本质又是一个枚举，内部分别是 Ok 和 Err，是 Ok 时则代表正常返回值，Err 则代表异常。

3，使用 ? 后，你不需要挨个判断并返回，任何一个 ? 返回 Err 了函数都会直接返回 Err。

4，unwrap 和 Option 的一样，正常则拿值，异常则 panic!

## 参考资料

- [rust入门秘籍-模式匹配的基本使用](https://rust-book.junmajinlong.com/ch10/01_pattern_match_basis.html)
- https://blog.vgot.net/archives/rust-some.html