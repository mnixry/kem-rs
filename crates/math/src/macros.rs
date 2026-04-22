include!(concat!(env!("OUT_DIR"), "/unroll.rs"));

#[cfg(test)]
mod tests {

    #[test]
    fn test_unroll_block() {
        let mut sum = 0;
        unroll!(i in (..5), {
            sum += i + 1;
        });
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_unroll_assign() {
        let (sum, result) = const {
            let mut sum = 0;
            let result = unroll!(i in [..5], {
                sum += i + 1;
                sum
            });
            (sum, result)
        };
        assert_eq!(sum, 15);
        assert_eq!(result, [1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_unroll() {
        let mut sum1 = 0;
        unroll!(i in (..5), {
            sum1 += i;
        });
        assert_eq!(sum1, 10);
        let mut sum2 = 0;
        unroll!(i in (..=5), {
            sum2 += i;
        });
        assert_eq!(sum2, 15);
    }
}
