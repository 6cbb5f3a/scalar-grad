#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <set>
#include <vector>
#include <print>
#include <cassert>

using f32 = float;
using f64 = double;

struct TensorImpl {
  f32 data;
  f32 grad;

  std::vector<std::shared_ptr<TensorImpl>> prev;
  std::function<void(f32)> grad_fn;
  TensorImpl(f32 data) : data(data), grad(0) {}
};

struct Tensor {
  std::shared_ptr<TensorImpl> impl;
  
  Tensor(f32 data) : impl(std::make_shared<TensorImpl>(data)) {}
  Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}

  f32 data() const { return impl->data; }
  f32 grad() const { return impl->grad; }

  void zero_grad() { impl->grad = 0; }

  void backward() {
    impl->grad = 1.0;
    
    std::vector<std::shared_ptr<TensorImpl>> path;
    std::set<TensorImpl*> vis;
    [&](this auto&& self, std::shared_ptr<TensorImpl> impl) -> void {
      vis.insert(impl.get());
      for (auto& ch : impl->prev) {
        if (ch->grad_fn != nullptr && vis.count(ch.get()) == 0) {
          self(ch);
        }
      }
      path.push_back(impl);
    } (impl);
    
    for (auto&& impl : path | std::views::reverse) {
      if (impl->grad_fn != nullptr) {
        impl->grad_fn(impl->grad);
      }
    }
  }
};

Tensor operator+(const Tensor& a, const Tensor& b) {
  Tensor c(a.data() + b.data());
  
  c.impl->prev.push_back(a.impl);
  c.impl->prev.push_back(b.impl);
  
  c.impl->grad_fn = [a, b](f32 grad) {
    a.impl->grad += grad;
    b.impl->grad += grad;
  };
  return c;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  Tensor c(a.data() - b.data());
  
  c.impl->prev.push_back(a.impl);
  c.impl->prev.push_back(b.impl);
  
  c.impl->grad_fn = [a, b](f32 grad) {
    a.impl->grad += grad;
    b.impl->grad -= grad;
  };
  return c;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  Tensor c(a.data() * b.data());
  
  c.impl->prev.push_back(a.impl);
  c.impl->prev.push_back(b.impl);
  
  c.impl->grad_fn = [a, b](f32 grad) {
    a.impl->grad += b.impl->data * grad;
    b.impl->grad += a.impl->data * grad;
  };
  return c;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
  assert(std::abs(b.data()) > std::numeric_limits<f32>::min() && "divide by zero");
  Tensor c(a.data() / b.data());
  
  c.impl->prev.push_back(a.impl);
  c.impl->prev.push_back(b.impl);

  c.impl->grad_fn = [a, b](f32 grad) {
    a.impl->grad += grad / b.data();
    b.impl->grad -= grad * a.data() / (b.data() * b.data());
  };
  return c;
}

Tensor pow(const Tensor& a, const Tensor& b) {
  Tensor c(std::pow(a.data(), b.data()));

  c.impl->prev.push_back(a.impl);
  c.impl->prev.push_back(b.impl);
  
  c.impl->grad_fn = [a, b](f32 grad) {
    a.impl->grad += grad * b.data() * std::pow(a.data(), b.data() - 1);
    b.impl->grad += grad * std::log(a.data()) * std::pow(a.data(), b.data());
  };

  return c;
}

// test memory leak
void test_case() {
  for (int i = 0; i < 5000000; ++i) {
      Tensor x(2.0);
      Tensor z(3.0);
      Tensor y = x * x + x;
      Tensor w = y / z;
      if (i % 10000 == 0) {
        std::println("step:{}", i);
      }
  }
}


Tensor op(Tensor x) {
  Tensor a = 2.0;
  Tensor b = 4.0;
  Tensor c = 8.0;
  x = x * a;
  x = a / x;
  x = x - b;
  x = x + c;
  x = x / 2;
  x = 7 / x;
  x = pow(x, 0.5);
  return x;
}

f64 op(f64 x) {
  f64 a = 2.0;
  f64 b = 4.0;
  f64 c = 8.0;
  x = x * a;
  x = a / x;
  x = x - b;
  x = x + c;
  x = x / 2;
  x = 7 / x;
  x = pow(x, 0.5);
  return x;
}

// check derivative by finite difference
void check(f32 x) {
  constexpr f64 h = 1e-5;
  constexpr f64 t = 1e-3;
  
  f32 actual = 0.0;
  Tensor w = x;
  Tensor y = op(w);
  y.backward();
  actual = w.grad();

  f64 expect = 0.0;
  f64 y0 = op(x + h);
  f64 y1 = op(x - h);
  expect = (y0 - y1) / (2 * h);
  if (abs(expect - actual) < t) {
    std::println("Accept");
    std::println("expect={:.5f}, actual={:.5f}", expect, actual);
  } else {
    std::println("Reject");
    std::println("expect={:.5f}, actual={:.5f}", expect, actual);
  }
}

int main(int argc, char* argv[]) {
  f64 x = 0.01;
  check(x);
  f64 grad = std::sqrt(7.0 / 2.0) * std::pow(1.0 / (1.0 / x + 4.0), 3.0 / 2.0) / (x * x);
  std::println("grad={:.5f}", grad);
  return 0;
}