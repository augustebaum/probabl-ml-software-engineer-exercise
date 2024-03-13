let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/9a5995e26c4da88f6927f2ce9409873f54f7a074";
in
  {pkgs ? import nixpkgs {}}: let
    python = pkgs.python311;
  in
    pkgs.mkShell {
      packages =
        [python]
        ++ (with python.pkgs; [
          scikit-learn
          matplotlib
          jupyterlab

          # dev
          ipython
          jupytext
          pip
          ipdb
        ]);
    }
