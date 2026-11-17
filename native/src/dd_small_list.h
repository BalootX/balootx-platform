#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace baloot::dd {
template<class T, std::size_t Capacity>
class small_list {
    std::array<T, Capacity> items{};
    std::size_t count = 0;
public:
    void push_back(T value) {
        if (count == Capacity) throw std::length_error("DD small list capacity exceeded");
        items.at(count) = value;
        ++count;
    }
    auto begin() { return items.begin(); }
    auto end() { return items.begin()+count; }
    std::size_t size() const { return count; }
    T operator[](std::size_t index) const {
        if (index >= count) throw std::out_of_range("DD small list index outside live size");
        return items.at(index);
    }
    template<class Less = std::less<T>>
    void sort(Less less = {}) {
        // These 8/16-element buffers need only live-size insertion sorting.
        for (std::size_t i = 1; i < count; ++i) {
            const auto value = items.at(i);
            auto j = i;
            while (j && less(value, items.at(j-1))) {
                items.at(j) = items.at(j-1);
                --j;
            }
            items.at(j) = value;
        }
    }
};
}
