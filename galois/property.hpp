#pragma once

#include <functional>
#include <optional>
#include <stdexcept>

namespace galois {

template <typename T>
class Property {
   public:
    std::optional<T> storedValue;
    std::function<T()> getter;
    std::function<void(T)> setter;

   public:
    // 存储值的构造函数
    Property() = default;
    Property(const T& value) : storedValue(value) {}

    // 访问器的构造函数
    Property(std::function<T()> get, std::function<void(T)> set = nullptr)
        : getter(std::move(get)), setter(std::move(set)) {}

    // 赋值运算符
    Property& operator=(const T& value) {
        if (setter) {
            setter(value);
        } else if (getter) {
            throw std::runtime_error("Property is read-only");
        } else {
            storedValue = value;
        }
        return *this;
    }

    // 指针访问运算符
    T operator->() {
        // Note: This creates a temporary, so only use for simple types or if you know what
        // you're doing. For a more robust solution, you may want to store the value in a
        // mutable member.
        T temp = getter();
        return temp;
    }

    // 类型转换运算符
    operator T() const {
        if (getter) {
            return getter();
        } else if (storedValue) {
            return *storedValue;
        }
        throw std::runtime_error("Property has no value");
    }

    // 获取值
    T get() const { return operator T(); }

    // 设置值
    void set(T value) { operator=(value); }
};

}  // namespace galois
