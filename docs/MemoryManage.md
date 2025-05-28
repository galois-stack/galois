# Memory Manage

我们的Tensor有多种形式, 我们在内存管理的时候主要考虑这些情况

## 申请释放

Alloc, 目前Alloc是唯一会申请内存的操作, 其内部相当于调用了c语言的malloc函数
Free, 释放存储的操作, 相当于c语言的free函数

## View操作

后缀带View的操作, 都是视图操作, 它不会会申请或者释放内存, 其通过改变数据的shape, stride等来改变数据的索引方式.

## Call操作

Call操作的input传入, 属于浅拷贝, 其应该增加引用计数
Return操作也属于浅拷贝
