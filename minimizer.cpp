template <typename valType>
class Minimizer {

    valType val;

public:
    size_t mmer_size;
    valType mask;

    Minimizer() = default;
    Minimizer(const Minimizer&) = default;
    Minimizer(Minimizer&&) = default;
    Minimizer& operator=(const Minimizer&) = default;
    Minimizer& operator=(Minimizer&&) = default;
    ~Minimizer() = default;

    void setMmerSize(const size_t mmer_size) {
        this->mmer_size = mmer_size;
        this->mask = (1 << (2 * mmer_size)) - 1;
    }

    valType getValue() const {
        return val;
    }

    template <baseIteratorType baseIter>
    void init(baseIter begin) {
        this->val = 0;
        for (size_t i = 0; i < mmer_size; ++i) {
            this->val <<= 2;
            this->val |= *begin;
            ++begin;
        }
    } 

    template <baseIteratorType baseIter>
    void next(baseIter begin) {
        this->val <<= 2;
        this->val &= mask;
        this->val |= *begin;
    }

    bool operator<(const Minimizer& other) const {
        return this->val < other.val;
    }

    bool operator==(const Minimizer& other) const {
        return this->val == other.val;
    }

    bool operator!=(const Minimizer& other) const {
        return this->val != other.val;
    }

    bool operator>(const Minimizer& other) const {
        return this->val > other.val;
    }
};
    