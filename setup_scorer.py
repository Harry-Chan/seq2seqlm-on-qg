import os

if __name__ == "__main__":
    # nlg-eval for `Our Scorer`
    os.system(
        "sudo apt-get -y update && sudo apt-get -y install default-jre && sudo apt-get -y install default-jdk"
    )
    os.system("export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup ${data_path}")
    import stanza

    stanza.download("en")
