Make sure all our code quality checks pass
1. format all code with `cargo fmt`
2. use `cargo lint` to check lint errors (do not use another lint command)
3. fix lint issues until `cargo lint` produces no errors
4. check format with `cargo fmt --check`
5. fix all format issues 
6. ensure there are no extra dependencies with `cargo machete`
7. ensure `cargo lint` is still clean
8. commit the cleanup, don't add claude as a co-author
