from line_profiler import LineProfiler

def profile_lines(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler_wrapper = profiler(func)
        result = profiler_wrapper(*args, **kwargs)

        print("\n=== Line-by-line profiling ===")
        profiler.print_stats()

        return result
    return wrapper

