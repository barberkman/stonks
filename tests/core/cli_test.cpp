#include <gtest/gtest.h>

#include "cli.h"

using stonks::app::wants_gui;

TEST(WantsGui, NoArgsIsHeadless) {
    const char* argv[] = { "app" };
    EXPECT_FALSE(wants_gui(1, argv));
}

TEST(WantsGui, GuiFlagSelectsGui) {
    const char* argv[] = { "app", "--gui" };
    EXPECT_TRUE(wants_gui(2, argv));
}

TEST(WantsGui, GuiFlagAmongOtherArgs) {
    const char* argv[] = { "app", "--verbose", "--gui", "extra" };
    EXPECT_TRUE(wants_gui(4, argv));
}

TEST(WantsGui, UnknownFlagIsHeadless) {
    const char* argv[] = { "app", "--nope" };
    EXPECT_FALSE(wants_gui(2, argv));
}
