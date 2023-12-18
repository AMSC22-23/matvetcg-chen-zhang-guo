A few comments:
 - you could consider archiving not maintained branches to avoid any misunderstanding (https://docs.github.com/en/repositories/archiving-a-github-repository/archiving-repositories), or at least setting the default branch to the current one (https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch)
 - it would be better to state in the readme that you use submodules, as well as to include only what you really need (or to explain it in the readme), because currently the repo cloned with --recursive is quite heavy:
```
~/matvetcg-chen-zhang-guo    main  du -sh .   
799M	.
```
 - more generally speaking, you could consider better specifying compilation instructions or testing portability of your code. For example, it can't find eigen unless "/usr/include/eigen3" is explicitely stated, and has some linking problem with mpi (open mpi 4.1.5). The usage of unsupported eigen features could be stated in the README.md too. 
