import higoal_cli


def main():
    m = higoal_cli.connect()
    for dev in higoal_cli.snapshot(m):
        for e in dev["entities"]:
            if e["type"] == "shutter":
                print(f"{dev['device']} | {e['name']} | {e['state']}")
    m.mq.stop()


if __name__ == "__main__":
    main()
